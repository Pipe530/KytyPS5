#include "common/assert.h"
#include "common/common.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "gpu_blit_shaders/gpu_blit_fs_triangle_spv.h"
#include "gpu_blit_shaders/gpu_fsr_easu_spv.h"
#include "gpu_blit_shaders/gpu_fsr_rcas_spv.h"
#include "gpu_blit_shaders/gpu_fxaa_spv.h"
#include "gpu_blit_shaders/gpu_smaa_blend_spv.h"
#include "gpu_blit_shaders/gpu_smaa_edges_spv.h"
#include "gpu_blit_shaders/gpu_smaa_weights_spv.h"
#include "gpu_blit_shaders/gpu_video_out_overlay_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/openxrHeadset.h"
#include "graphics/presentation/performanceOverlay.h"
#include "graphics/presentation/presenter.h"
#include "graphics/presentation/systemOverlay.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window/windowInternal.h"
#include "graphics/presentation/window/smaa/AreaTex.h"
#include "graphics/presentation/window/smaa/SearchTex.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>
#include <vulkan/vk_platform.h>

// IWYU pragma: no_include <intrin.h>

namespace Libs::Graphics {

struct Presenter::Frame {
	VulkanImage   image;
	vk::ImageView view         = nullptr;
	uint64_t      present_tick = 0;
	bool          busy         = false;
	bool          hdr          = false;
	bool          blank        = false;

	void Configure(GraphicContext& graphics, vk::Extent2D extent, vk::Format format);
	void Transit(vk::CommandBuffer command, vk::ImageLayout layout, vk::AccessFlags2 access);
	void CopyFrom(CommandBuffer& command, Image& source);
	void Clear(CommandBuffer& command, const vk::ClearColorValue& color);
};

class FramePool final {
public:
	FramePool(WindowContext& window, CommandScheduler& scheduler)
	    : m_window(window), m_scheduler(scheduler) {}
	~FramePool() {
		m_scheduler.Wait(m_scheduler.CurrentTick() - 1);
		for (auto& frame: m_frames) {
			m_window.graphic_ctx.device.destroyImageView(frame->view, nullptr);
			if (frame->image.image != nullptr) {
				m_window.graphic_ctx.DeleteImage(frame->image);
			}
		}
	}
	KYTY_CLASS_NO_COPY(FramePool);

	void Initialize(uint32_t count, vk::Format format) {
		if (count == 0 || format == vk::Format::eUndefined) {
			EXIT("prepared-frame pool requires at least one frame\n");
		}
		Common::LockGuard lock(m_mutex);
		if (!m_frames.empty()) {
			EXIT("prepared-frame pool was initialized twice\n");
		}
		m_format = format;
		m_frames.reserve(count);
		m_free.reserve(count);
		for (uint32_t i = 0; i < count; i++) {
			auto frame = std::make_unique<Presenter::Frame>();
			m_free.push_back(frame.get());
			m_frames.push_back(std::move(frame));
		}
	}

	void SetFormat(vk::Format format) {
		if (format == vk::Format::eUndefined) {
			EXIT("prepared-frame pool requires a presentation format\n");
		}
		Common::LockGuard lock(m_mutex);
		m_format = format;
	}

	vk::Format GetFormat() {
		Common::LockGuard lock(m_mutex);
		if (m_format == vk::Format::eUndefined) {
			EXIT("prepared-frame pool has no presentation format\n");
		}
		return m_format;
	}

	Presenter::Frame* Acquire(vk::Extent2D extent, vk::Format format) {
		m_mutex.Lock();
		if (m_frames.empty()) {
			EXIT("prepared-frame pool was used before swapchain initialization\n");
		}
		// A synchronized flip may need more frames than the swapchain has images.
		// Queue capacity bounds growth; waiting here can prevent its master from recording.
		if (m_free.empty()) {
			auto frame = std::make_unique<Presenter::Frame>();
			m_free.push_back(frame.get());
			m_frames.push_back(std::move(frame));
		}
		const auto compatible = std::ranges::find_if(m_free, [&](const auto* frame) {
			return frame->image.extent.width == extent.width &&
			       frame->image.extent.height == extent.height && frame->image.format == format;
		});
		auto*      frame      = compatible == m_free.end() ? m_free.back() : *compatible;
		if (compatible != m_free.end()) {
			*compatible = m_free.back();
		}
		m_free.pop_back();
		if (frame->busy) {
			EXIT("prepared-frame pool returned an invalid frame\n");
		}
		frame->busy = true;
		m_mutex.Unlock();

		m_scheduler.Wait(frame->present_tick);
		return frame;
	}

	void ValidateForPresent(Presenter::Frame* frame) {
		Common::LockGuard lock(m_mutex);
		if (frame == nullptr || !frame->busy) {
			EXIT("prepared frame has invalid presentation ownership\n");
		}
	}

	void Release(Presenter::Frame* frame) {
		if (frame == nullptr) {
			EXIT("cannot release a null prepared frame\n");
		}
		Common::LockGuard lock(m_mutex);
		if (!frame->busy) {
			EXIT("prepared frame was released twice\n");
		}
		frame->busy = false;
		m_free.push_back(frame);
	}

private:
	WindowContext&                                 m_window;
	CommandScheduler&                              m_scheduler;
	Common::Mutex                                  m_mutex;
	std::vector<std::unique_ptr<Presenter::Frame>> m_frames;
	std::vector<Presenter::Frame*>                 m_free;
	vk::Format                                     m_format = vk::Format::eUndefined;
};

void Presenter::Frame::Configure(GraphicContext& graphics, vk::Extent2D extent, vk::Format format) {
	if (extent.width == 0 || extent.height == 0 || format == vk::Format::eUndefined) {
		EXIT("unsupported prepared frame, extent=%ux%u format=%d\n", extent.width, extent.height,
		     static_cast<int>(format));
	}
	auto&      dst        = image;
	const bool compatible = dst.image != nullptr && dst.extent.width == extent.width &&
	                        dst.extent.height == extent.height && dst.format == format;
	if (compatible) {
		return;
	}

	const auto features = graphics.GetFormatProperties(format).optimalTilingFeatures;
	const auto required =
	    vk::FormatFeatureFlagBits::eBlitSrc | vk::FormatFeatureFlagBits::eSampledImageFilterLinear |
	    vk::FormatFeatureFlagBits::eSampledImage | vk::FormatFeatureFlagBits::eTransferSrc |
	    vk::FormatFeatureFlagBits::eTransferDst;
	if ((features & required) != required) {
		EXIT("prepared presentation format lacks optimal blit support: format=%d features=0x%x\n",
		     static_cast<int>(format), static_cast<vk::FormatFeatureFlags::MaskType>(features));
	}

	if (dst.image != nullptr) {
		graphics.device.destroyImageView(view, nullptr);
		view = nullptr;
		graphics.DeleteImage(dst);
	}

	vk::ImageCreateInfo create {};
	create.sType         = vk::StructureType::eImageCreateInfo;
	create.imageType     = vk::ImageType::e2D;
	create.extent        = {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.format        = format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
	               vk::ImageUsageFlagBits::eSampled;
	create.sharingMode = vk::SharingMode::eExclusive;
	create.samples     = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create, dst)) {
		EXIT("failed to allocate prepared presentation image, extent=%ux%u format=%d\n",
		     extent.width, extent.height, static_cast<int>(format));
	}
}

void Presenter::Frame::Transit(vk::CommandBuffer command, vk::ImageLayout layout,
                               vk::AccessFlags2 access) {
	const auto     stage  = access == vk::AccessFlagBits2::eTransferRead ||
	                                access == vk::AccessFlagBits2::eTransferWrite
	                            ? vk::PipelineStageFlagBits2::eTransfer
	                            : vk::PipelineStageFlagBits2::eAllCommands;
	constexpr auto writes = vk::AccessFlagBits2::eTransferWrite |
	                        vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eMemoryWrite;
	if (image.state.layout == layout && image.state.access_mask == access &&
	    !static_cast<bool>(image.state.access_mask & writes)) {
		return;
	}
	vk::ImageMemoryBarrier2 barrier {};
	barrier.srcStageMask                    = image.state.pl_stage;
	barrier.srcAccessMask                   = image.state.access_mask;
	barrier.dstStageMask                    = stage;
	barrier.dstAccessMask                   = access;
	barrier.oldLayout                       = image.state.layout;
	barrier.newLayout                       = layout;
	barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	barrier.image                           = image.image;
	barrier.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
	barrier.subresourceRange.baseMipLevel   = 0;
	barrier.subresourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
	image.state = {stage, access, layout};
	image.subresource_states.clear();
}

void Presenter::Frame::CopyFrom(CommandBuffer& command_buffer, Image& source) {
	command_buffer.EndRendering();
	auto command = command_buffer.Handle();
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
	               command);
	Transit(command, vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite);
	vk::ImageCopy copy {};
	copy.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, source.backing.layers};
	copy.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, image.layers};
	copy.extent         = {std::min(source.backing.extent.width, image.extent.width),
	                       std::min(source.backing.extent.height, image.extent.height), 1};
	EXIT_IF(copy.srcSubresource.layerCount != copy.dstSubresource.layerCount);
	command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, image.image,
	                  vk::ImageLayout::eTransferDstOptimal, copy);
}

void Presenter::Frame::Clear(CommandBuffer& command_buffer, const vk::ClearColorValue& color) {
	command_buffer.EndRendering();
	auto command = command_buffer.Handle();
	Transit(command, vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite);
	const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	command.clearColorImage(image.image, vk::ImageLayout::eTransferDstOptimal, &color, 1, &range);
}

class Swapchain final {
public:
	enum class Status : uint8_t { Success, Recreate, SurfaceLost };

	explicit Swapchain(WindowContext& window): m_window(window) {}
	~Swapchain();
	KYTY_CLASS_NO_COPY(Swapchain);

	void                 Create();
	void                 Recreate(bool surface_lost = false);
	[[nodiscard]] bool   NeedsResize() const;
	[[nodiscard]] Status AcquireNextImage(CommandScheduler& scheduler);
	[[nodiscard]] bool   PrepareSystemOverlay();
	void     RecordPresentCommands(CommandBuffer& command, Presenter::Frame* source,
	                               const Presenter::Layer& overlay, bool draw_system_overlay);
	uint64_t Submit(CommandScheduler& scheduler);
	[[nodiscard]] Status Present();

	[[nodiscard]] uint32_t ImageCount() const noexcept {
		return static_cast<uint32_t>(m_images.size());
	}
	[[nodiscard]] vk::Format Format() const noexcept { return m_format; }
	[[nodiscard]] bool IsHdrSupported() const noexcept { return m_hdr_supported.load(); }
	bool SetHdr(bool hdr) {
		const bool changed = m_hdr_active != (hdr && IsHdrSupported());
		m_hdr_requested = hdr;
		return changed;
	}

private:
	vk::SurfaceFormatKHR SelectFormat();
	void Destroy();
	void DrawOverlay(vk::CommandBuffer command, const Presenter::Layer& layer);
	[[nodiscard]] bool UseFsr(const Presenter::Frame* source) const;
	void EnsureFsrResources();
	void EnsureFxaaImage(vk::Extent2D extent);
	struct SmaaImage {
		VulkanImage   image {};
		vk::ImageView view = nullptr;
	};
	void CreateSmaaImage(SmaaImage& out, vk::Format format, vk::Extent2D extent,
	                     vk::ImageUsageFlags usage);
	void DestroySmaaImage(SmaaImage& image);
	void EnsureSmaaTargets(vk::Extent2D extent);
	void EnsureSmaaLookup();
	void RecordSmaaLookupUpload(vk::CommandBuffer command);
	void RecordFsr(vk::CommandBuffer command, Presenter::Frame& source, bool keep_attachment);
	void DestroyFsr();
	WindowContext&   m_window;
	vk::SwapchainKHR m_handle = nullptr;
	vk::Format       m_format = vk::Format::eUndefined;
	std::atomic_bool m_hdr_supported {false};
	bool             m_hdr_requested = false;
	bool             m_hdr_active    = false;
	vk::Extent2D     m_extent {};
	// Drawable pixel size observed when this swapchain was created.
	vk::Extent2D                   m_window_extent {};
	std::vector<vk::Image>         m_images;
	std::vector<vk::ImageView>     m_image_views;
	std::vector<vk::Semaphore>     m_image_acquired;
	std::vector<uint64_t>          m_image_acquired_ticks;
	std::vector<vk::Semaphore>     m_render_complete;
	std::unique_ptr<SystemOverlay> m_system_overlay;
	vk::DescriptorSetLayout        m_overlay_descriptors = nullptr;
	vk::PipelineLayout             m_overlay_layout      = nullptr;
	vk::Pipeline                   m_overlay_pipeline    = nullptr;
	vk::Sampler                    m_overlay_sampler     = nullptr;
	vk::DescriptorSetLayout        m_fsr_descriptors     = nullptr;
	vk::PipelineLayout             m_fsr_layout          = nullptr;
	vk::Pipeline                   m_fsr_easu_pipeline   = nullptr;
	vk::Pipeline                   m_fsr_rcas_pipeline   = nullptr;
	vk::Pipeline                   m_fxaa_pipeline       = nullptr;
	vk::Sampler                    m_fxaa_sampler        = nullptr;
	VulkanImage                    m_fxaa_image {};
	vk::ImageView                  m_fxaa_view           = nullptr;
	vk::Extent2D                   m_fxaa_extent {};
	vk::Pipeline                   m_smaa_edges_pipeline   = nullptr;
	vk::Pipeline                   m_smaa_weights_pipeline = nullptr;
	vk::Pipeline                   m_smaa_blend_pipeline   = nullptr;
	SmaaImage                      m_smaa_edges {};
	SmaaImage                      m_smaa_weights {};
	vk::Extent2D                   m_smaa_extent {};
	SmaaImage                      m_smaa_area {};
	SmaaImage                      m_smaa_search {};
	vk::Buffer                     m_smaa_staging            = nullptr;
	VmaAllocation                  m_smaa_staging_allocation = nullptr;
	bool                           m_smaa_lookup_uploaded    = false;	vk::Sampler                    m_fsr_sampler         = nullptr;
	VulkanImage                    m_fsr_image {};
	vk::ImageView                  m_fsr_view            = nullptr;
	uint32_t                       m_image_index         = static_cast<uint32_t>(-1);
	uint32_t                       m_frame_index         = 0;
};

struct Presenter::Impl {
	explicit Impl(WindowContext& owner)
	    : renderer(*owner.render_context), window(owner), swapchain(owner),
	      present_scheduler(renderer, owner.graphic_ctx), frames(owner, present_scheduler) {
		EXIT_IF(owner.render_context == nullptr);
		swapchain.Create();
		frames.Initialize(swapchain.ImageCount(), swapchain.Format());
	}

	void RecoverSwapchain(Swapchain::Status status) {
		LOGF("Recovering Vulkan swapchain%s\n",
		     status == Swapchain::Status::SurfaceLost ? " and surface" : "");
		Common::DebugCounters::Add(Common::DebugCounters::Counter::SwapchainRecreations);
		swapchain.Recreate(status == Swapchain::Status::SurfaceLost);
		frames.SetFormat(swapchain.Format());
	}

	Image& ResolveSurface(const ImageInfo& info) {
		TextureCache::ImageDesc desc {};
		desc.info                  = info;
		desc.view_info.format      = info.pixel_format;
		desc.view_info.type        = vk::ImageViewType::e2D;
		desc.view_info.aspect      = vk::ImageAspectFlagBits::eColor;
		desc.view_info.base_level  = 0;
		desc.view_info.level_count = 1;
		desc.view_info.base_layer  = 0;
		desc.view_info.layer_count = 1;
		desc.view_info.usage       = vk::ImageUsageFlagBits::eTransferSrc;
		desc.type                  = TextureCache::BindingType::VideoOut;

		auto&      cache      = renderer.GetTextureCache();
		const auto image_id   = cache.FindImage(desc);
		auto&      image      = cache.GetImage(image_id);
		image.usage.video_out = true;
		cache.UpdateImage(image_id);
		return image;
	}
	void Present();

	RenderContext&        renderer;
	WindowContext&        window;
	Swapchain             swapchain;
	CommandScheduler      present_scheduler;
	FramePool             frames;
	Common::Mutex         present_mutex;
	std::array<Layer, 2>  layers {};
	std::atomic<uint64_t> presented_overlay_revision {0};
};

vk::SurfaceFormatKHR Swapchain::SelectFormat() {
	const auto& formats    = m_window.surface_capabilities.formats;
	auto&       graphics   = m_window.graphic_ctx;
	const auto  hdr_format = std::ranges::find_if(formats, [&](const auto& candidate) {
		return candidate.colorSpace == vk::ColorSpaceKHR::eHdr10St2084EXT &&
		       IsHdrSwapchainFormat(candidate.format) &&
		       static_cast<bool>(
		           graphics.GetFormatProperties(candidate.format).optimalTilingFeatures &
		           vk::FormatFeatureFlagBits::eBlitDst);
	});
	m_hdr_supported        = Config::HdrEnabled() && hdr_format != formats.end();
	m_hdr_active           = m_hdr_requested && IsHdrSupported();
	if (m_hdr_active) {
		return *hdr_format;
	}
	if (Config::HdrEnabled() && !IsHdrSupported()) {
		LOGF("HDR10 surface unavailable; using SDR output\n");
	}
	if (formats.size() == 1 && formats.front().format == vk::Format::eUndefined) {
		return {vk::Format::eR8G8B8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear};
	}
	for (const auto& candidate: formats) {
		if (candidate.colorSpace != vk::ColorSpaceKHR::eSrgbNonlinear) {
			continue;
		}
		switch (candidate.format) {
			case vk::Format::eB8G8R8A8Unorm:
			case vk::Format::eR8G8B8A8Unorm: return candidate;
			default: break;
		}
	}
	EXIT("no supported UNORM swapchain format\n");
}

void Swapchain::Create() {
	auto& graphics = m_window.graphic_ctx;
	EXIT_IF(graphics.device == nullptr);
	EXIT_IF(m_window.surface == nullptr);

	// Capture the size before querying surface capabilities. If the window changes
	// during creation, the next presentation will detect the newer size.
	{
		Common::LockGuard lock(m_window.mutex);
		m_window_extent = {graphics.screen_width, graphics.screen_height};
	}
	EXIT_IF(m_window_extent.width == 0);
	EXIT_IF(m_window_extent.height == 0);
	m_window.RefreshSurfaceCapabilities();
	const auto& surface = m_window.surface_capabilities;
	EXIT_NOT_IMPLEMENTED(surface.formats.empty());

	m_extent = surface.capabilities.currentExtent;
	if (m_extent.width == std::numeric_limits<uint32_t>::max()) {
		m_extent.width =
		    std::clamp(m_window_extent.width, surface.capabilities.minImageExtent.width,
		               surface.capabilities.maxImageExtent.width);
		m_extent.height =
		    std::clamp(m_window_extent.height, surface.capabilities.minImageExtent.height,
		               surface.capabilities.maxImageExtent.height);
	}
	uint32_t image_count = surface.capabilities.minImageCount + 1;
	if (surface.capabilities.maxImageCount != 0) {
		image_count = std::min(image_count, surface.capabilities.maxImageCount);
	}
	const auto transform =
	    surface.capabilities.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity
	        ? vk::SurfaceTransformFlagBitsKHR::eIdentity
	        : surface.capabilities.currentTransform;
	const auto composite =
	    surface.capabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::eOpaque
	        ? vk::CompositeAlphaFlagBitsKHR::eOpaque
	        : vk::CompositeAlphaFlagBitsKHR::eInherit;

	const auto format = SelectFormat();
	m_format                      = format.format;
	const auto swapchain_features = graphics.GetFormatProperties(m_format).optimalTilingFeatures;
	if (!static_cast<bool>(swapchain_features & vk::FormatFeatureFlagBits::eBlitDst)) {
		EXIT("swapchain format cannot be a blit destination: format=%d\n",
		     static_cast<int>(m_format));
	}

	vk::SwapchainCreateInfoKHR create_info {};
	create_info.sType            = vk::StructureType::eSwapchainCreateInfoKHR;
	create_info.surface          = m_window.surface;
	create_info.minImageCount    = image_count;
	create_info.imageFormat      = format.format;
	create_info.imageColorSpace  = format.colorSpace;
	create_info.imageExtent      = m_extent;
	create_info.imageArrayLayers = 1;
	create_info.imageUsage =
	    vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst;
	create_info.imageSharingMode = vk::SharingMode::eExclusive;
	create_info.preTransform     = transform;
	create_info.compositeAlpha   = composite;
	switch (Config::GetPresentMode()) {
		case Config::PresentMode::Mailbox:
			create_info.presentMode = vk::PresentModeKHR::eMailbox;
			break;
		case Config::PresentMode::Immediate:
			create_info.presentMode = vk::PresentModeKHR::eImmediate;
			break;
		case Config::PresentMode::Fifo:
		default: create_info.presentMode = vk::PresentModeKHR::eFifo; break;
	}
	if (std::find(surface.present_modes.begin(), surface.present_modes.end(),
	              create_info.presentMode) == surface.present_modes.end()) {
		LOGF("warning: requested present mode is unavailable; falling back to Fifo\n");
		create_info.presentMode = vk::PresentModeKHR::eFifo;
	}
	create_info.clipped = VK_TRUE;
	RequireVulkanSuccess(graphics.device.createSwapchainKHR(&create_info, nullptr, &m_handle),
	                     "vkCreateSwapchainKHR");
	EXIT_IF(m_handle == nullptr);
	LOGF("Vulkan presentation: %s\n", m_hdr_active ? "HDR10 (PQ)" : "SDR");

	m_images = EnumerateVulkan<vk::Image>(
	    "vkGetSwapchainImagesKHR", [&](uint32_t* count, vk::Image* images) {
		    return graphics.device.getSwapchainImagesKHR(m_handle, count, images);
	    });
	EXIT_NOT_IMPLEMENTED(m_images.empty());

	m_image_views.resize(m_images.size());
	for (size_t i = 0; i < m_images.size(); i++) {
		vk::ImageViewCreateInfo view {};
		view.sType                           = vk::StructureType::eImageViewCreateInfo;
		view.image                           = m_images[i];
		view.viewType                        = vk::ImageViewType::e2D;
		view.format                          = m_format;
		view.components                      = {};
		view.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
		view.subresourceRange.baseArrayLayer = 0;
		view.subresourceRange.baseMipLevel   = 0;
		view.subresourceRange.layerCount     = 1;
		view.subresourceRange.levelCount     = 1;
		RequireVulkanSuccess(graphics.device.createImageView(&view, nullptr, &m_image_views[i]),
		                     "vkCreateImageView");
		EXIT_IF(m_image_views[i] == nullptr);
	}

	vk::SemaphoreCreateInfo semaphore_info {};
	semaphore_info.sType = vk::StructureType::eSemaphoreCreateInfo;
	m_image_acquired.resize(m_images.size());
	m_image_acquired_ticks.assign(m_images.size(), 0);
	m_render_complete.resize(m_images.size());
	for (size_t i = 0; i < m_images.size(); i++) {
		RequireVulkanSuccess(
		    graphics.device.createSemaphore(&semaphore_info, nullptr, &m_image_acquired[i]),
		    "create swapchain image-acquired semaphore");
		RequireVulkanSuccess(
		    graphics.device.createSemaphore(&semaphore_info, nullptr, &m_render_complete[i]),
		    "create swapchain render-complete semaphore");
	}
	m_image_index = static_cast<uint32_t>(-1);
	m_frame_index = 0;
}

Swapchain::~Swapchain() {
	Destroy();
}

void Swapchain::Destroy() {
	if (m_handle == nullptr && m_image_acquired.empty() && m_render_complete.empty() &&
	    m_image_views.empty()) {
		return;
	}
	auto& graphics = m_window.graphic_ctx;

	{
		Common::LockGuard queue_lock(graphics.queue_mutex);
		RequireVulkanSuccess(graphics.queue.waitIdle(), "wait for swapchain queue");
	}
	if (m_system_overlay != nullptr) {
		m_system_overlay->ReleaseVulkan();
	}
	graphics.device.destroyPipeline(m_overlay_pipeline, nullptr);
	graphics.device.destroyPipelineLayout(m_overlay_layout, nullptr);
	graphics.device.destroyDescriptorSetLayout(m_overlay_descriptors, nullptr);
	graphics.device.destroySampler(m_overlay_sampler, nullptr);
	m_overlay_pipeline    = nullptr;
	m_overlay_layout      = nullptr;
	m_overlay_descriptors = nullptr;
	m_overlay_sampler     = nullptr;
	DestroyFsr();

	for (const auto semaphore: m_image_acquired) {
		if (semaphore != nullptr) {
			graphics.device.destroySemaphore(semaphore, nullptr);
		}
	}
	for (const auto semaphore: m_render_complete) {
		if (semaphore != nullptr) {
			graphics.device.destroySemaphore(semaphore, nullptr);
		}
	}
	for (const auto view: m_image_views) {
		if (view != nullptr) {
			graphics.device.destroyImageView(view, nullptr);
		}
	}
	if (m_handle != nullptr) {
		graphics.device.destroySwapchainKHR(m_handle, nullptr);
	}

	m_handle        = nullptr;
	m_format        = vk::Format::eUndefined;
	m_extent        = {};
	m_window_extent = {};
	m_image_index   = static_cast<uint32_t>(-1);
	m_frame_index   = 0;
	m_images.clear();
	m_image_views.clear();
	m_image_acquired.clear();
	m_image_acquired_ticks.clear();
	m_render_complete.clear();
}

void Swapchain::Recreate(bool surface_lost) {
	Destroy();
	if (surface_lost) {
#if defined(__APPLE__)
		// Surface recreation goes through SDL_Vulkan_CreateSurface, which touches the
		// window's view/layer and must run on the main thread on macOS.
		EXIT_IF(!SDL_RunOnMainThread(
		    [](void* window) { static_cast<WindowContext*>(window)->RecreateSurface(); }, &m_window,
		    true));
#else
		m_window.RecreateSurface();
#endif
	}
	Create();
}

bool Swapchain::NeedsResize() const {
	Common::LockGuard lock(m_window.mutex);
	return m_window_extent.width != m_window.graphic_ctx.screen_width ||
	       m_window_extent.height != m_window.graphic_ctx.screen_height;
}

Swapchain::Status Swapchain::AcquireNextImage(CommandScheduler& scheduler) {
	EXIT_IF(m_handle == nullptr || m_frame_index >= m_image_acquired.size());
	// Repeated frames can wrap the semaphore ring without acquiring a new source frame.
	// Wait for the submission that consumed this slot before signaling it again.
	scheduler.Wait(m_image_acquired_ticks[m_frame_index]);
	m_image_index     = static_cast<uint32_t>(-1);
	const auto result = m_window.graphic_ctx.device.acquireNextImageKHR(
	    m_handle, std::numeric_limits<uint64_t>::max(), m_image_acquired[m_frame_index], nullptr,
	    &m_image_index);
	switch (result) {
		case vk::Result::eSuccess: break;
		case vk::Result::eSuboptimalKHR:
			LOGF("vkAcquireNextImageKHR returned vk::Result::eSuboptimalKHR\n");
			return Status::Recreate;
		case vk::Result::eErrorOutOfDateKHR:
			LOGF("vkAcquireNextImageKHR returned vk::Result::eErrorOutOfDateKHR\n");
			return Status::Recreate;
		case vk::Result::eErrorUnknown:
			LOGF("vkAcquireNextImageKHR returned vk::Result::eErrorUnknown\n");
			return Status::Recreate;
		case vk::Result::eErrorSurfaceLostKHR:
			LOGF("vkAcquireNextImageKHR returned vk::Result::eErrorSurfaceLostKHR\n");
			return Status::SurfaceLost;
		default: EXIT("vkAcquireNextImageKHR failed: %s\n", vk::to_string(result).c_str());
	}
	EXIT_IF(m_image_index >= m_images.size());
	return Status::Success;
}

bool Swapchain::PrepareSystemOverlay() {
	if (m_system_overlay == nullptr) {
		m_system_overlay = std::make_unique<SystemOverlay>(m_window.graphic_ctx);
	}
	return m_system_overlay->PrepareFrame(m_extent, m_format, ImageCount());
}

void Swapchain::DrawOverlay(vk::CommandBuffer command, const Presenter::Layer& layer) {
	auto device = m_window.graphic_ctx.device;
	if (m_overlay_pipeline == nullptr) {
		vk::SamplerCreateInfo sampler {};
		sampler.magFilter    = vk::Filter::eLinear;
		sampler.minFilter    = vk::Filter::eLinear;
		sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
		RequireVulkanSuccess(device.createSampler(&sampler, nullptr, &m_overlay_sampler),
		                     "create video-out overlay sampler");
		const vk::DescriptorSetLayoutBinding binding {0, vk::DescriptorType::eCombinedImageSampler,
		                                              1, vk::ShaderStageFlagBits::eFragment};
		vk::DescriptorSetLayoutCreateInfo    descriptors {};
		descriptors.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
		descriptors.bindingCount = 1;
		descriptors.pBindings    = &binding;
		RequireVulkanSuccess(
		    device.createDescriptorSetLayout(&descriptors, nullptr, &m_overlay_descriptors),
		    "create video-out overlay descriptor layout");
		const vk::PushConstantRange alpha {vk::ShaderStageFlagBits::eFragment, 0, sizeof(uint32_t)};
		vk::PipelineLayoutCreateInfo layout {};
		layout.setLayoutCount         = 1;
		layout.pSetLayouts            = &m_overlay_descriptors;
		layout.pushConstantRangeCount = 1;
		layout.pPushConstantRanges    = &alpha;
		RequireVulkanSuccess(device.createPipelineLayout(&layout, nullptr, &m_overlay_layout),
		                     "create video-out overlay pipeline layout");
		const auto vertex   = CompileSPV(GPU_BLIT_FS_TRIANGLE_SPV, device);
		const auto fragment = CompileSPV(GPU_VIDEO_OUT_OVERLAY_SPV, device);
		std::array<vk::PipelineShaderStageCreateInfo, 2> stages {};
		stages[0].stage  = vk::ShaderStageFlagBits::eVertex;
		stages[0].module = vertex;
		stages[0].pName  = "main";
		stages[1].stage  = vk::ShaderStageFlagBits::eFragment;
		stages[1].module = fragment;
		stages[1].pName  = "main";
		vk::PipelineVertexInputStateCreateInfo   vertex_input {};
		vk::PipelineInputAssemblyStateCreateInfo assembly {};
		assembly.topology = vk::PrimitiveTopology::eTriangleList;
		vk::PipelineViewportStateCreateInfo viewport {};
		viewport.viewportCount = 1;
		viewport.scissorCount  = 1;
		vk::PipelineRasterizationStateCreateInfo rasterization {};
		rasterization.lineWidth = 1.0f;
		vk::PipelineMultisampleStateCreateInfo multisample {};
		multisample.rasterizationSamples = vk::SampleCountFlagBits::e1;
		vk::PipelineColorBlendAttachmentState attachment {};
		attachment.blendEnable         = VK_TRUE;
		attachment.srcColorBlendFactor = vk::BlendFactor::eOne;
		attachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
		attachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
		attachment.dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
		attachment.colorWriteMask = vk::ColorComponentFlagBits::eR |
		                            vk::ColorComponentFlagBits::eG |
		                            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
		vk::PipelineColorBlendStateCreateInfo blend {};
		blend.attachmentCount = 1;
		blend.pAttachments    = &attachment;
		const std::array dynamic_states {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
		vk::PipelineDynamicStateCreateInfo dynamic {};
		dynamic.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
		dynamic.pDynamicStates    = dynamic_states.data();
		vk::PipelineRenderingCreateInfo rendering {};
		rendering.colorAttachmentCount    = 1;
		rendering.pColorAttachmentFormats = &m_format;
		vk::GraphicsPipelineCreateInfo create {};
		create.pNext               = &rendering;
		create.stageCount          = static_cast<uint32_t>(stages.size());
		create.pStages             = stages.data();
		create.pVertexInputState   = &vertex_input;
		create.pInputAssemblyState = &assembly;
		create.pViewportState      = &viewport;
		create.pRasterizationState = &rasterization;
		create.pMultisampleState   = &multisample;
		create.pColorBlendState    = &blend;
		create.pDynamicState       = &dynamic;
		create.layout              = m_overlay_layout;
		RequireVulkanSuccess(
		    device.createGraphicsPipelines(nullptr, 1, &create, nullptr, &m_overlay_pipeline),
		    "create video-out overlay pipeline");
		device.destroyShaderModule(fragment, nullptr);
		device.destroyShaderModule(vertex, nullptr);
	}
	auto& frame = *layer.frame;
	if (frame.view == nullptr) {
		vk::ImageViewCreateInfo view {};
		view.image            = frame.image.image;
		view.viewType         = vk::ImageViewType::e2D;
		view.format           = frame.image.format;
		view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		RequireVulkanSuccess(device.createImageView(&view, nullptr, &frame.view),
		                     "create video-out overlay image view");
	}
	const vk::DescriptorImageInfo image {m_overlay_sampler, frame.view,
	                                     vk::ImageLayout::eShaderReadOnlyOptimal};
	vk::WriteDescriptorSet        write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
	write.pImageInfo      = &image;
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, m_overlay_layout, 0, 1, &write);
	const uint32_t flags = (layer.premultiplied_alpha ? 1u : 0u) |
	                       (m_hdr_active ? 2u : 0u) | (frame.hdr ? 4u : 0u) |
	                       (frame.image.format == vk::Format::eR16G16B16A16Sfloat ? 8u : 0u);
	command.pushConstants(m_overlay_layout, vk::ShaderStageFlagBits::eFragment, 0,
	                      sizeof(flags), &flags);
	command.bindPipeline(vk::PipelineBindPoint::eGraphics, m_overlay_pipeline);
	vk::RenderingAttachmentInfo attachment {};
	attachment.imageView   = m_image_views[m_image_index];
	attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	attachment.loadOp      = vk::AttachmentLoadOp::eLoad;
	attachment.storeOp     = vk::AttachmentStoreOp::eStore;
	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = m_extent;
	rendering.layerCount           = 1;
	rendering.colorAttachmentCount = 1;
	rendering.pColorAttachments    = &attachment;
	command.beginRendering(&rendering);
	const vk::Viewport viewport {
	    0.0f, 0.0f, static_cast<float>(m_extent.width), static_cast<float>(m_extent.height),
	    0.0f, 1.0f};
	const vk::Rect2D scissor {{0, 0}, m_extent};
	command.setViewport(0, 1, &viewport);
	command.setScissor(0, 1, &scissor);
	command.draw(3, 1, 0, 0);
	command.endRendering();
}

bool Swapchain::UseFsr(const Presenter::Frame* source) const {
	if (!Config::FsrEnabled() || source == nullptr || source->blank) {
		return false;
	}
	const auto extent = source->image.extent;
	// FSR 1.0 only upscales; linear float sources are not in the perceptual space it expects.
	if (source->image.format == vk::Format::eR16G16B16A16Sfloat && !source->hdr) {
		return false;
	}
	return extent.width <= m_extent.width && extent.height <= m_extent.height &&
	       (extent.width < m_extent.width || extent.height < m_extent.height);
}

void Swapchain::DestroyFsr() {
	auto& graphics = m_window.graphic_ctx;
	auto  device   = graphics.device;
	device.destroyPipeline(m_fxaa_pipeline, nullptr);
	device.destroyPipeline(m_smaa_edges_pipeline, nullptr);
	device.destroyPipeline(m_smaa_weights_pipeline, nullptr);
	device.destroyPipeline(m_smaa_blend_pipeline, nullptr);
	m_smaa_edges_pipeline   = nullptr;
	m_smaa_weights_pipeline = nullptr;
	m_smaa_blend_pipeline   = nullptr;
	DestroySmaaImage(m_smaa_edges);
	DestroySmaaImage(m_smaa_weights);
	DestroySmaaImage(m_smaa_area);
	DestroySmaaImage(m_smaa_search);
	m_smaa_extent = vk::Extent2D {};
	if (m_smaa_staging != nullptr) {
		vmaDestroyBuffer(graphics.allocator, static_cast<VkBuffer>(m_smaa_staging),
		                 m_smaa_staging_allocation);
		m_smaa_staging            = nullptr;
		m_smaa_staging_allocation = nullptr;
	}
	m_smaa_lookup_uploaded = false;
	device.destroyPipeline(m_fxaa_pipeline, nullptr);
	device.destroySampler(m_fxaa_sampler, nullptr);
	device.destroyImageView(m_fxaa_view, nullptr);
	if (m_fxaa_image.image != nullptr) {
		graphics.DeleteImage(m_fxaa_image);
	}
	m_fxaa_pipeline = nullptr;
	m_fxaa_sampler  = nullptr;
	m_fxaa_view     = nullptr;
	m_fxaa_extent   = vk::Extent2D {};
	device.destroyPipeline(m_fsr_easu_pipeline, nullptr);
	device.destroyPipeline(m_fsr_rcas_pipeline, nullptr);
	device.destroyPipelineLayout(m_fsr_layout, nullptr);
	device.destroyDescriptorSetLayout(m_fsr_descriptors, nullptr);
	device.destroySampler(m_fsr_sampler, nullptr);
	device.destroyImageView(m_fsr_view, nullptr);
	if (m_fsr_image.image != nullptr) {
		graphics.DeleteImage(m_fsr_image);
	}
	m_fsr_easu_pipeline = nullptr;
	m_fsr_rcas_pipeline = nullptr;
	m_fsr_layout        = nullptr;
	m_fsr_descriptors   = nullptr;
	m_fsr_sampler       = nullptr;
	m_fsr_view          = nullptr;
}

void Swapchain::EnsureFsrResources() {
	if (m_fsr_easu_pipeline != nullptr) {
		return;
	}
	auto& graphics = m_window.graphic_ctx;
	auto  device   = graphics.device;

	constexpr auto fsr_format = vk::Format::eR16G16B16A16Sfloat;
	vk::ImageCreateInfo create_image {};
	create_image.imageType     = vk::ImageType::e2D;
	create_image.extent        = {m_extent.width, m_extent.height, 1};
	create_image.mipLevels     = 1;
	create_image.arrayLayers   = 1;
	create_image.format        = fsr_format;
	create_image.tiling        = vk::ImageTiling::eOptimal;
	create_image.initialLayout = vk::ImageLayout::eUndefined;
	create_image.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;
	create_image.sharingMode = vk::SharingMode::eExclusive;
	create_image.samples     = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create_image, m_fsr_image)) {
		EXIT("failed to allocate FSR intermediate image, extent=%ux%u\n", m_extent.width,
		     m_extent.height);
	}
	vk::ImageViewCreateInfo view {};
	view.image            = m_fsr_image.image;
	view.viewType         = vk::ImageViewType::e2D;
	view.format           = fsr_format;
	view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	RequireVulkanSuccess(device.createImageView(&view, nullptr, &m_fsr_view),
	                     "create FSR intermediate image view");

	vk::SamplerCreateInfo sampler {};
	sampler.magFilter    = vk::Filter::eNearest;
	sampler.minFilter    = vk::Filter::eNearest;
	sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
	sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
	sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
	RequireVulkanSuccess(device.createSampler(&sampler, nullptr, &m_fsr_sampler),
	                     "create FSR sampler");
	const std::array<vk::DescriptorSetLayoutBinding, 3> bindings {{
	    {0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment},
	    {1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment},
	    {2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment},
	}};
	vk::DescriptorSetLayoutCreateInfo descriptors {};
	descriptors.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	descriptors.bindingCount = static_cast<uint32_t>(bindings.size());
	descriptors.pBindings    = bindings.data();
	RequireVulkanSuccess(device.createDescriptorSetLayout(&descriptors, nullptr, &m_fsr_descriptors),
	                     "create FSR descriptor layout");
	const vk::PushConstantRange params {vk::ShaderStageFlagBits::eFragment, 0, 5 * sizeof(float)};
	vk::PipelineLayoutCreateInfo layout {};
	layout.setLayoutCount         = 1;
	layout.pSetLayouts            = &m_fsr_descriptors;
	layout.pushConstantRangeCount = 1;
	layout.pPushConstantRanges    = &params;
	RequireVulkanSuccess(device.createPipelineLayout(&layout, nullptr, &m_fsr_layout),
	                     "create FSR pipeline layout");

	const auto vertex = CompileSPV(GPU_BLIT_FS_TRIANGLE_SPV, device);
	auto make_pipeline = [&](const auto& fragment_spv, vk::Format target, vk::Pipeline& out) {
		const auto fragment = CompileSPV(fragment_spv, device);
		std::array<vk::PipelineShaderStageCreateInfo, 2> stages {};
		stages[0].stage  = vk::ShaderStageFlagBits::eVertex;
		stages[0].module = vertex;
		stages[0].pName  = "main";
		stages[1].stage  = vk::ShaderStageFlagBits::eFragment;
		stages[1].module = fragment;
		stages[1].pName  = "main";
		vk::PipelineVertexInputStateCreateInfo   vertex_input {};
		vk::PipelineInputAssemblyStateCreateInfo assembly {};
		assembly.topology = vk::PrimitiveTopology::eTriangleList;
		vk::PipelineViewportStateCreateInfo viewport {};
		viewport.viewportCount = 1;
		viewport.scissorCount  = 1;
		vk::PipelineRasterizationStateCreateInfo rasterization {};
		rasterization.lineWidth = 1.0f;
		vk::PipelineMultisampleStateCreateInfo multisample {};
		multisample.rasterizationSamples = vk::SampleCountFlagBits::e1;
		vk::PipelineColorBlendAttachmentState attachment {};
		attachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
		                            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
		vk::PipelineColorBlendStateCreateInfo blend {};
		blend.attachmentCount = 1;
		blend.pAttachments    = &attachment;
		const std::array dynamic_states {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
		vk::PipelineDynamicStateCreateInfo dynamic {};
		dynamic.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
		dynamic.pDynamicStates    = dynamic_states.data();
		vk::PipelineRenderingCreateInfo rendering {};
		rendering.colorAttachmentCount    = 1;
		rendering.pColorAttachmentFormats = &target;
		vk::GraphicsPipelineCreateInfo create {};
		create.pNext               = &rendering;
		create.stageCount          = static_cast<uint32_t>(stages.size());
		create.pStages             = stages.data();
		create.pVertexInputState   = &vertex_input;
		create.pInputAssemblyState = &assembly;
		create.pViewportState      = &viewport;
		create.pRasterizationState = &rasterization;
		create.pMultisampleState   = &multisample;
		create.pColorBlendState    = &blend;
		create.pDynamicState       = &dynamic;
		create.layout              = m_fsr_layout;
		RequireVulkanSuccess(device.createGraphicsPipelines(nullptr, 1, &create, nullptr, &out),
		                     "create FSR pipeline");
		device.destroyShaderModule(fragment, nullptr);
	};
	make_pipeline(GPU_FSR_RCAS_SPV, m_format, m_fsr_rcas_pipeline);
	make_pipeline(GPU_FSR_EASU_SPV, fsr_format, m_fsr_easu_pipeline);
	make_pipeline(GPU_FXAA_SPV, fsr_format, m_fxaa_pipeline);
	make_pipeline(GPU_SMAA_EDGES_SPV, vk::Format::eR8G8Unorm, m_smaa_edges_pipeline);
	make_pipeline(GPU_SMAA_WEIGHTS_SPV, vk::Format::eR8G8B8A8Unorm, m_smaa_weights_pipeline);
	make_pipeline(GPU_SMAA_BLEND_SPV, fsr_format, m_smaa_blend_pipeline);
	device.destroyShaderModule(vertex, nullptr);

	vk::SamplerCreateInfo linear = sampler;
	linear.magFilter             = vk::Filter::eLinear;
	linear.minFilter             = vk::Filter::eLinear;
	RequireVulkanSuccess(device.createSampler(&linear, nullptr, &m_fxaa_sampler),
	                     "create FXAA sampler");
}

void Swapchain::EnsureFxaaImage(vk::Extent2D extent) {
	if (m_fxaa_image.image != nullptr && m_fxaa_extent == extent) {
		return;
	}
	auto& graphics = m_window.graphic_ctx;
	auto  device   = graphics.device;
	if (m_fxaa_image.image != nullptr) {
		device.waitIdle();
		device.destroyImageView(m_fxaa_view, nullptr);
		m_fxaa_view = nullptr;
		graphics.DeleteImage(m_fxaa_image);
	}
	vk::ImageCreateInfo create_image {};
	create_image.imageType     = vk::ImageType::e2D;
	create_image.extent        = {extent.width, extent.height, 1};
	create_image.mipLevels     = 1;
	create_image.arrayLayers   = 1;
	create_image.format        = vk::Format::eR16G16B16A16Sfloat;
	create_image.tiling        = vk::ImageTiling::eOptimal;
	create_image.initialLayout = vk::ImageLayout::eUndefined;
	create_image.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;
	create_image.sharingMode = vk::SharingMode::eExclusive;
	create_image.samples     = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create_image, m_fxaa_image)) {
		EXIT("failed to allocate FXAA image, extent=%ux%u\n", extent.width, extent.height);
	}
	vk::ImageViewCreateInfo view {};
	view.image            = m_fxaa_image.image;
	view.viewType         = vk::ImageViewType::e2D;
	view.format           = vk::Format::eR16G16B16A16Sfloat;
	view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	RequireVulkanSuccess(device.createImageView(&view, nullptr, &m_fxaa_view),
	                     "create FXAA image view");
	m_fxaa_extent = extent;
}

void Swapchain::CreateSmaaImage(SmaaImage& out, vk::Format format, vk::Extent2D extent,
                                vk::ImageUsageFlags usage) {
	auto& graphics = m_window.graphic_ctx;
	vk::ImageCreateInfo create_image {};
	create_image.imageType     = vk::ImageType::e2D;
	create_image.extent        = {extent.width, extent.height, 1};
	create_image.mipLevels     = 1;
	create_image.arrayLayers   = 1;
	create_image.format        = format;
	create_image.tiling        = vk::ImageTiling::eOptimal;
	create_image.initialLayout = vk::ImageLayout::eUndefined;
	create_image.usage         = usage | vk::ImageUsageFlagBits::eSampled;
	create_image.sharingMode   = vk::SharingMode::eExclusive;
	create_image.samples       = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create_image, out.image)) {
		EXIT("failed to allocate SMAA image, extent=%ux%u\n", extent.width, extent.height);
	}
	vk::ImageViewCreateInfo view {};
	view.image            = out.image.image;
	view.viewType         = vk::ImageViewType::e2D;
	view.format           = format;
	view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	RequireVulkanSuccess(graphics.device.createImageView(&view, nullptr, &out.view),
	                     "create SMAA image view");
}

void Swapchain::DestroySmaaImage(SmaaImage& image) {
	auto& graphics = m_window.graphic_ctx;
	graphics.device.destroyImageView(image.view, nullptr);
	image.view = nullptr;
	if (image.image.image != nullptr) {
		graphics.DeleteImage(image.image);
	}
}

void Swapchain::EnsureSmaaTargets(vk::Extent2D extent) {
	if (m_smaa_edges.image.image != nullptr && m_smaa_extent == extent) {
		return;
	}
	if (m_smaa_edges.image.image != nullptr) {
		m_window.graphic_ctx.device.waitIdle();
		DestroySmaaImage(m_smaa_edges);
		DestroySmaaImage(m_smaa_weights);
	}
	constexpr auto usage = vk::ImageUsageFlagBits::eColorAttachment;
	CreateSmaaImage(m_smaa_edges, vk::Format::eR8G8Unorm, extent, usage);
	CreateSmaaImage(m_smaa_weights, vk::Format::eR8G8B8A8Unorm, extent, usage);
	m_smaa_extent = extent;
}

void Swapchain::EnsureSmaaLookup() {
	if (m_smaa_staging != nullptr) {
		return;
	}
	auto& graphics = m_window.graphic_ctx;
	CreateSmaaImage(m_smaa_area, vk::Format::eR8G8Unorm, {AREATEX_WIDTH, AREATEX_HEIGHT},
	                vk::ImageUsageFlagBits::eTransferDst);
	CreateSmaaImage(m_smaa_search, vk::Format::eR8Unorm, {SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT},
	                vk::ImageUsageFlagBits::eTransferDst);

	static_assert(sizeof(areaTexBytes) == AREATEX_SIZE);
	static_assert(sizeof(searchTexBytes) == SEARCHTEX_SIZE);
	vk::BufferCreateInfo buffer {};
	buffer.size  = AREATEX_SIZE + SEARCHTEX_SIZE;
	buffer.usage = vk::BufferUsageFlagBits::eTransferSrc;
	VmaAllocationCreateInfo allocation {};
	allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	allocation.flags =
	    VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
	VmaAllocationInfo mapped {};
	VkBuffer          handle = VK_NULL_HANDLE;
	if (vmaCreateBuffer(graphics.allocator, static_cast<const VkBufferCreateInfo*>(buffer),
	                    &allocation, &handle, &m_smaa_staging_allocation, &mapped) != VK_SUCCESS) {
		EXIT("failed to allocate SMAA lookup upload buffer\n");
	}
	m_smaa_staging = vk::Buffer(handle);
	auto* data     = static_cast<uint8_t*>(mapped.pMappedData);
	std::memcpy(data, areaTexBytes, AREATEX_SIZE);
	std::memcpy(data + AREATEX_SIZE, searchTexBytes, SEARCHTEX_SIZE);
	RequireVulkanSuccess(static_cast<vk::Result>(vmaFlushAllocation(
	                         graphics.allocator, m_smaa_staging_allocation, 0, VK_WHOLE_SIZE)),
	                     "flush SMAA lookup upload");
}

void Swapchain::RecordSmaaLookupUpload(vk::CommandBuffer command) {
	if (m_smaa_lookup_uploaded) {
		return;
	}
	EnsureSmaaLookup();
	constexpr vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	auto transition = [&](vk::Image image, vk::PipelineStageFlags2 src_stage,
	                      vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst_stage,
	                      vk::AccessFlags2 dst_access, vk::ImageLayout old_layout,
	                      vk::ImageLayout new_layout) {
		vk::ImageMemoryBarrier2 b {};
		b.srcStageMask        = src_stage;
		b.srcAccessMask       = src_access;
		b.dstStageMask        = dst_stage;
		b.dstAccessMask       = dst_access;
		b.oldLayout           = old_layout;
		b.newLayout           = new_layout;
		b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.image               = image;
		b.subresourceRange    = range;
		vk::DependencyInfo dependency {};
		dependency.imageMemoryBarrierCount = 1;
		dependency.pImageMemoryBarriers    = &b;
		command.pipelineBarrier2(dependency);
	};
	auto upload = [&](SmaaImage& image, vk::DeviceSize offset) {
		transition(image.image.image, vk::PipelineStageFlagBits2::eTopOfPipe, {},
		           vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
		           vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
		vk::BufferImageCopy copy {};
		copy.bufferOffset     = offset;
		copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		copy.imageExtent      = image.image.extent;
		command.copyBufferToImage(m_smaa_staging, image.image.image,
		                          vk::ImageLayout::eTransferDstOptimal, 1, &copy);
		transition(image.image.image, vk::PipelineStageFlagBits2::eTransfer,
		           vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eFragmentShader,
		           vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eTransferDstOptimal,
		           vk::ImageLayout::eShaderReadOnlyOptimal);
	};
	upload(m_smaa_area, 0);
	upload(m_smaa_search, AREATEX_SIZE);
	m_smaa_lookup_uploaded = true;
}

void Swapchain::RecordFsr(vk::CommandBuffer command, Presenter::Frame& source,
                          bool keep_attachment) {
	EnsureFsrResources();
	auto device = m_window.graphic_ctx.device;

	if (source.view == nullptr) {
		vk::ImageViewCreateInfo view {};
		view.image            = source.image.image;
		view.viewType         = vk::ImageViewType::e2D;
		view.format           = source.image.format;
		view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		RequireVulkanSuccess(device.createImageView(&view, nullptr, &source.view),
		                     "create FSR source image view");
	}
	source.Transit(command, vk::ImageLayout::eShaderReadOnlyOptimal,
	               vk::AccessFlagBits2::eShaderRead);

	constexpr vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	auto barrier = [&](vk::Image image, vk::PipelineStageFlags2 src_stage,
	                   vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst_stage,
	                   vk::AccessFlags2 dst_access, vk::ImageLayout old_layout,
	                   vk::ImageLayout new_layout) {
		vk::ImageMemoryBarrier2 b {};
		b.srcStageMask        = src_stage;
		b.srcAccessMask       = src_access;
		b.dstStageMask        = dst_stage;
		b.dstAccessMask       = dst_access;
		b.oldLayout           = old_layout;
		b.newLayout           = new_layout;
		b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.image               = image;
		b.subresourceRange    = range;
		vk::DependencyInfo dependency {};
		dependency.imageMemoryBarrierCount = 1;
		dependency.pImageMemoryBarriers    = &b;
		command.pipelineBarrier2(dependency);
	};

	// The intermediate image is fully rewritten each frame. The swapchain barrier uses the
	// transfer stage to chain after the acquire semaphore wait.
	barrier(m_fsr_image.image, vk::PipelineStageFlagBits2::eFragmentShader, {},
	        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
	        vk::AccessFlagBits2::eColorAttachmentWrite, vk::ImageLayout::eUndefined,
	        vk::ImageLayout::eColorAttachmentOptimal);
	barrier(m_images[m_image_index], vk::PipelineStageFlagBits2::eTransfer, {},
	        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
	        vk::AccessFlagBits2::eColorAttachmentWrite, vk::ImageLayout::eUndefined,
	        vk::ImageLayout::eColorAttachmentOptimal);

	const bool use_smaa = Config::SmaaEnabled();
	const bool use_fxaa = Config::FxaaEnabled() && !use_smaa;
	if (use_smaa) {
		RecordSmaaLookupUpload(command);
		EnsureSmaaTargets({source.image.extent.width, source.image.extent.height});
		for (auto* target : {&m_smaa_edges, &m_smaa_weights}) {
			barrier(target->image.image, vk::PipelineStageFlagBits2::eFragmentShader, {},
			        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
			        vk::AccessFlagBits2::eColorAttachmentWrite, vk::ImageLayout::eUndefined,
			        vk::ImageLayout::eColorAttachmentOptimal);
		}
	}
	if (use_fxaa || use_smaa) {
		EnsureFxaaImage({source.image.extent.width, source.image.extent.height});
		barrier(m_fxaa_image.image, vk::PipelineStageFlagBits2::eFragmentShader, {},
		        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		        vk::AccessFlagBits2::eColorAttachmentWrite, vk::ImageLayout::eUndefined,
		        vk::ImageLayout::eColorAttachmentOptimal);
	}

	struct Params {
		float in_size[2];
		float out_size[2];
		float sharpness;
	};
	// RCAS sharpness is configured in [0, 1]; map it to stops (0 stops = maximum, 2 = softest).
	const float  rcas_stops = (1.0f - Config::GetFsrSharpness()) * 2.0f;
	const Params fxaa {{static_cast<float>(source.image.extent.width),
	                    static_cast<float>(source.image.extent.height)},
	                   {static_cast<float>(source.image.extent.width),
	                    static_cast<float>(source.image.extent.height)},
	                   0.0f};
	const Params easu {{static_cast<float>(source.image.extent.width),
	                    static_cast<float>(source.image.extent.height)},
	                   {static_cast<float>(m_extent.width), static_cast<float>(m_extent.height)},
	                   0.0f};
	const Params rcas {{static_cast<float>(m_extent.width), static_cast<float>(m_extent.height)},
	                   {static_cast<float>(m_extent.width), static_cast<float>(m_extent.height)},
	                   std::exp2(-rcas_stops)};

	auto pass = [&](vk::Pipeline pipeline, std::initializer_list<vk::ImageView> inputs,
	                vk::Sampler input_sampler, vk::ImageView target, vk::Extent2D extent,
	                const Params& params) {
		std::array<vk::DescriptorImageInfo, 3> images {};
		std::array<vk::WriteDescriptorSet, 3>  writes {};
		uint32_t                               count = 0;
		for (const auto input : inputs) {
			images[count] = {input_sampler, input, vk::ImageLayout::eShaderReadOnlyOptimal};
			writes[count].dstBinding      = count;
			writes[count].descriptorCount = 1;
			writes[count].descriptorType  = vk::DescriptorType::eCombinedImageSampler;
			writes[count].pImageInfo      = &images[count];
			count++;
		}
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, m_fsr_layout, 0, count,
		                             writes.data());
		command.pushConstants(m_fsr_layout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(params),
		                      &params);
		command.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
		vk::RenderingAttachmentInfo attachment {};
		attachment.imageView   = target;
		attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
		attachment.loadOp      = vk::AttachmentLoadOp::eDontCare;
		attachment.storeOp     = vk::AttachmentStoreOp::eStore;
		vk::RenderingInfo rendering {};
		rendering.renderArea.extent    = extent;
		rendering.layerCount           = 1;
		rendering.colorAttachmentCount = 1;
		rendering.pColorAttachments    = &attachment;
		command.beginRendering(&rendering);
		const vk::Viewport viewport {0.0f, 0.0f, static_cast<float>(extent.width),
		                             static_cast<float>(extent.height), 0.0f, 1.0f};
		const vk::Rect2D   scissor {{0, 0}, extent};
		command.setViewport(0, 1, &viewport);
		command.setScissor(0, 1, &scissor);
		command.draw(3, 1, 0, 0);
		command.endRendering();
	};

	auto easu_input = source.view;
	if (use_smaa) {
		auto to_read = [&](vk::Image image) {
			barrier(image, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
			        vk::AccessFlagBits2::eColorAttachmentWrite,
			        vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderRead,
			        vk::ImageLayout::eColorAttachmentOptimal,
			        vk::ImageLayout::eShaderReadOnlyOptimal);
		};
		pass(m_smaa_edges_pipeline, {source.view}, m_fxaa_sampler, m_smaa_edges.view,
		     m_smaa_extent, fxaa);
		to_read(m_smaa_edges.image.image);
		pass(m_smaa_weights_pipeline, {m_smaa_edges.view, m_smaa_area.view, m_smaa_search.view},
		     m_fxaa_sampler, m_smaa_weights.view, m_smaa_extent, fxaa);
		to_read(m_smaa_weights.image.image);
		pass(m_smaa_blend_pipeline, {source.view, m_smaa_weights.view}, m_fxaa_sampler,
		     m_fxaa_view, m_fxaa_extent, fxaa);
		to_read(m_fxaa_image.image);
		easu_input = m_fxaa_view;
	} else if (use_fxaa) {
		pass(m_fxaa_pipeline, {source.view}, m_fxaa_sampler, m_fxaa_view, m_fxaa_extent, fxaa);
		barrier(m_fxaa_image.image, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		        vk::AccessFlagBits2::eColorAttachmentWrite,
		        vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderRead,
		        vk::ImageLayout::eColorAttachmentOptimal,
		        vk::ImageLayout::eShaderReadOnlyOptimal);
		easu_input = m_fxaa_view;
	}
	pass(m_fsr_easu_pipeline, {easu_input}, m_fsr_sampler, m_fsr_view, m_extent, easu);
	barrier(m_fsr_image.image, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
	        vk::AccessFlagBits2::eColorAttachmentWrite, vk::PipelineStageFlagBits2::eFragmentShader,
	        vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eColorAttachmentOptimal,
	        vk::ImageLayout::eShaderReadOnlyOptimal);
	pass(m_fsr_rcas_pipeline, {m_fsr_view}, m_fsr_sampler, m_image_views[m_image_index], m_extent,
	     rcas);

	// Later overlay passes load the swapchain image; otherwise it goes straight to presentation.
	barrier(m_images[m_image_index], vk::PipelineStageFlagBits2::eColorAttachmentOutput,
	        vk::AccessFlagBits2::eColorAttachmentWrite,
	        keep_attachment ? vk::PipelineStageFlagBits2::eColorAttachmentOutput
	                        : vk::PipelineStageFlagBits2::eAllCommands,
	        keep_attachment ? vk::AccessFlagBits2::eColorAttachmentRead |
	                              vk::AccessFlagBits2::eColorAttachmentWrite
	                        : vk::AccessFlagBits2::eMemoryRead,
	        vk::ImageLayout::eColorAttachmentOptimal,
	        keep_attachment ? vk::ImageLayout::eColorAttachmentOptimal
	                        : vk::ImageLayout::ePresentSrcKHR);
}

void Swapchain::RecordPresentCommands(CommandBuffer& command, Presenter::Frame* source,
                                      const Presenter::Layer& overlay, bool draw_system_overlay) {
	EXIT_IF(m_image_index >= m_images.size());
	auto       vk_command      = command.Handle();
	const bool draw_overlay    = overlay.frame != nullptr;
	const bool draw_attachment = draw_overlay || draw_system_overlay;
	const bool use_fsr         = UseFsr(source);
	if (source != nullptr && !use_fsr) {
		source->Transit(vk_command, vk::ImageLayout::eTransferSrcOptimal,
		                vk::AccessFlagBits2::eTransferRead);
	}
	if (draw_overlay) {
		overlay.frame->Transit(vk_command, vk::ImageLayout::eShaderReadOnlyOptimal,
		                       vk::AccessFlagBits2::eShaderRead);
	}

	vk::ImageMemoryBarrier to_transfer {};
	to_transfer.sType                           = vk::StructureType::eImageMemoryBarrier;
	to_transfer.dstAccessMask                   = vk::AccessFlagBits::eTransferWrite;
	to_transfer.oldLayout                       = vk::ImageLayout::eUndefined;
	to_transfer.newLayout                       = vk::ImageLayout::eTransferDstOptimal;
	to_transfer.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	to_transfer.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	to_transfer.image                           = m_images[m_image_index];
	to_transfer.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
	to_transfer.subresourceRange.baseMipLevel   = 0;
	to_transfer.subresourceRange.levelCount     = 1;
	to_transfer.subresourceRange.baseArrayLayer = 0;
	to_transfer.subresourceRange.layerCount     = 1;
	if (!use_fsr) {
		// Match the acquire wait stage so the layout transition cannot precede acquisition.
		vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                           vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags {}, 0,
		                           nullptr, 0, nullptr, 1, &to_transfer);
	}

	if (use_fsr) {
		RecordFsr(vk_command, *source, draw_attachment);
	} else if (source != nullptr) {
		vk::ImageBlit region {};
		region.srcSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
		region.srcSubresource.mipLevel       = 0;
		region.srcSubresource.baseArrayLayer = 0;
		region.srcSubresource.layerCount     = 1;
		region.srcOffsets[1].x               = static_cast<int>(source->image.extent.width);
		region.srcOffsets[1].y               = static_cast<int>(source->image.extent.height);
		region.srcOffsets[1].z               = 1;
		region.dstSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
		region.dstSubresource.mipLevel       = 0;
		region.dstSubresource.baseArrayLayer = 0;
		region.dstSubresource.layerCount     = 1;
		region.dstOffsets[1].x               = static_cast<int>(m_extent.width);
		region.dstOffsets[1].y               = static_cast<int>(m_extent.height);
		region.dstOffsets[1].z               = 1;
		vk_command.blitImage(source->image.image, vk::ImageLayout::eTransferSrcOptimal,
		                     m_images[m_image_index], vk::ImageLayout::eTransferDstOptimal, 1,
		                     &region, vk::Filter::eLinear);
	} else {
		const vk::ClearColorValue black {};
		vk_command.clearColorImage(m_images[m_image_index], vk::ImageLayout::eTransferDstOptimal,
		                           &black, 1, &to_transfer.subresourceRange);
	}

	vk::ImageMemoryBarrier to_present {};
	to_present.sType               = vk::StructureType::eImageMemoryBarrier;
	to_present.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
	to_present.dstAccessMask       = draw_attachment ? vk::AccessFlagBits::eColorAttachmentRead |
	                                                       vk::AccessFlagBits::eColorAttachmentWrite
	                                                 : vk::AccessFlagBits::eMemoryRead;
	to_present.oldLayout           = vk::ImageLayout::eTransferDstOptimal;
	to_present.newLayout           = draw_attachment ? vk::ImageLayout::eColorAttachmentOptimal
	                                                 : vk::ImageLayout::ePresentSrcKHR;
	to_present.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_present.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_present.image               = m_images[m_image_index];
	to_present.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
	to_present.subresourceRange.baseMipLevel   = 0;
	to_present.subresourceRange.levelCount     = 1;
	to_present.subresourceRange.baseArrayLayer = 0;
	to_present.subresourceRange.layerCount     = 1;
	if (!use_fsr) {
		vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                           draw_attachment ? vk::PipelineStageFlagBits::eColorAttachmentOutput
		                                           : vk::PipelineStageFlagBits::eAllCommands,
		                           vk::DependencyFlagBits::eByRegion, 0, nullptr, 0, nullptr, 1,
		                           &to_present);
	}
	if (draw_attachment) {
		if (draw_overlay) {
			DrawOverlay(vk_command, overlay);
			if (draw_system_overlay) {
				to_present.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
				to_present.oldLayout     = vk::ImageLayout::eColorAttachmentOptimal;
				vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
				                           vk::PipelineStageFlagBits::eColorAttachmentOutput,
				                           vk::DependencyFlagBits::eByRegion, 0, nullptr, 0,
				                           nullptr, 1, &to_present);
			}
		}
		if (draw_system_overlay) {
			m_system_overlay->Record(vk_command, m_image_views[m_image_index]);
		}
		to_present.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
		to_present.dstAccessMask = vk::AccessFlagBits::eMemoryRead;
		to_present.oldLayout     = vk::ImageLayout::eColorAttachmentOptimal;
		to_present.newLayout     = vk::ImageLayout::ePresentSrcKHR;
		vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
		                           vk::PipelineStageFlagBits::eAllCommands,
		                           vk::DependencyFlagBits::eByRegion, 0, nullptr, 0, nullptr, 1,
		                           &to_present);
	}
}

uint64_t Swapchain::Submit(CommandScheduler& scheduler) {
	EXIT_IF(m_frame_index >= m_image_acquired.size() || m_image_index >= m_render_complete.size());
	SubmitInfo submit;
	submit.AddWait(m_image_acquired[m_frame_index], 1, vk::PipelineStageFlagBits::eTransfer);
	submit.AddSignal(m_render_complete[m_image_index]);
	const auto tick                       = scheduler.Submit(submit);
	m_image_acquired_ticks[m_frame_index] = tick;
	return tick;
}

Swapchain::Status Swapchain::Present() {
	EXIT_IF(m_image_index >= m_render_complete.size());
	const auto         ready = m_render_complete[m_image_index];
	vk::PresentInfoKHR present {};
	present.sType              = vk::StructureType::ePresentInfoKHR;
	present.swapchainCount     = 1;
	present.pSwapchains        = &m_handle;
	present.pImageIndices      = &m_image_index;
	present.pWaitSemaphores    = &ready;
	present.waitSemaphoreCount = 1;

	vk::Result result;
	{
		Common::LockGuard lock(m_window.graphic_ctx.queue_mutex);
		result = m_window.graphic_ctx.queue.presentKHR(&present);
	}
	switch (result) {
		case vk::Result::eSuccess: break;
		case vk::Result::eSuboptimalKHR:
			LOGF("vkQueuePresentKHR returned vk::Result::eSuboptimalKHR\n");
			return Status::Recreate;
		case vk::Result::eErrorOutOfDateKHR:
			LOGF("vkQueuePresentKHR returned vk::Result::eErrorOutOfDateKHR\n");
			return Status::Recreate;
		case vk::Result::eErrorSurfaceLostKHR:
			LOGF("vkQueuePresentKHR returned vk::Result::eErrorSurfaceLostKHR\n");
			return Status::SurfaceLost;
		default: EXIT("vkQueuePresentKHR failed: %s\n", vk::to_string(result).c_str());
	}
	m_frame_index = (m_frame_index + 1u) % static_cast<uint32_t>(m_images.size());
	return Status::Success;
}

Presenter::Presenter(WindowContext& window): m_impl(std::make_unique<Impl>(window)) {}

Presenter::~Presenter() = default;

Presenter::Frame& Presenter::PrepareFrame(CommandBuffer& buffer, const ImageInfo& info, bool hdr,
                                          bool headset) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(buffer.IsInvalid());
	auto frame_format = info.pixel_format;
	switch (frame_format) {
		case vk::Format::eR8G8B8A8Srgb: frame_format = vk::Format::eR8G8B8A8Unorm; break;
		case vk::Format::eB8G8R8A8Srgb: frame_format = vk::Format::eB8G8R8A8Unorm; break;
		default: break;
	}
	vk::Extent2D frame_extent {info.extent.width, info.extent.height};
	if (Config::GetRenderScale() < 1.0f) {
		// A render-scaled surface is smaller than its guest extent; pick a matching frame.
		Common::LockGuard render_lock(m_impl->renderer.GetMutex());
		const auto&       probe = m_impl->ResolveSurface(info);
		frame_extent = {probe.backing.extent.width, probe.backing.extent.height};
	}
	auto* frame = m_impl->frames.Acquire(frame_extent, frame_format);
	frame->hdr   = hdr;
	frame->blank = false;
	Common::LockGuard render_lock(m_impl->renderer.GetMutex());
	auto&             image = m_impl->ResolveSurface(info);
	if (image.backing.format == vk::Format::eUndefined) {
		EXIT("unsupported presentation source, image=%p\n", static_cast<const void*>(&image));
	}
	{
		static int logged = 0;
		if (logged++ < 3) {
			LOGF("RenderScale: present info=%ux%u backing=%ux%u cached_info=%ux%u frame=%ux%u\n",
			     info.extent.width, info.extent.height, image.backing.extent.width,
			     image.backing.extent.height, image.info.extent.width, image.info.extent.height,
			     frame_extent.width, frame_extent.height);
		}
	}
	frame->Configure(m_impl->window.graphic_ctx,
	                 {image.backing.extent.width, image.backing.extent.height}, frame_format);
	frame->CopyFrom(buffer, image);
	if (headset) {
		// The frame holds the raw display-encoded pixels the headset needs.
		frame->Transit(buffer.Handle(), vk::ImageLayout::eTransferSrcOptimal,
		               vk::AccessFlagBits2::eTransferRead);
		Vr::OpenXrQueueEyeSource(buffer, frame->image);
	}
	return *frame;
}

Presenter::Frame& Presenter::PrepareBlankFrame(uint32_t width, uint32_t height, bool opaque,
                                               CommandBuffer* producer) {
	KYTY_PROFILER_FUNCTION();
	auto              format = m_impl->frames.GetFormat();
	auto*             frame  = m_impl->frames.Acquire({width, height}, format);
	frame->hdr   = false;
	frame->blank = true;
	frame->Configure(m_impl->window.graphic_ctx, {width, height}, format);
	vk::ClearColorValue clear {};
	clear.float32[3] = opaque ? 1.0f : 0.0f;
	if (producer != nullptr) {
		EXIT_IF(producer->IsInvalid());
		frame->Clear(*producer, clear);
	} else {
		Common::LockGuard lock(m_impl->present_mutex);
		auto& command = m_impl->present_scheduler.BeginCommand();
		frame->Clear(command, clear);
		frame->present_tick = m_impl->present_scheduler.Submit();
	}
	return *frame;
}

bool Presenter::PresentLastFrame() {
	Common::LockGuard lock(m_impl->present_mutex);
	if (m_impl->layers[0].frame == nullptr && m_impl->layers[1].frame == nullptr) {
		return false;
	}
	m_impl->Present();
	return true;
}

bool Presenter::IsGuestPaused() const noexcept {
	return m_impl->window.loop.paused.load(std::memory_order_acquire);
}

bool Presenter::IsHdrSupported() const noexcept {
	return m_impl->swapchain.IsHdrSupported();
}

bool Presenter::NeedsSystemOverlayRefresh() const noexcept {
	const auto visual = GetSystemOverlayVisualState();
	return visual.active ||
	       visual.revision != m_impl->presented_overlay_revision.load(std::memory_order_acquire);
}

RenderContext& Presenter::Renderer() const noexcept {
	return m_impl->renderer;
}

void Presenter::Present(Frame& frame) {
	const Layer layer {&frame, 0, false};
	Present(std::span(&layer, 1));
}

void Presenter::Present(std::span<const Layer> layers) {
	Common::LockGuard lock(m_impl->present_mutex);
	for (const auto& layer: layers) {
		EXIT_IF(layer.bus < 0 || layer.bus >= static_cast<int>(m_impl->layers.size()));
		m_impl->frames.ValidateForPresent(layer.frame);
		auto& previous = m_impl->layers[layer.bus];
		if (previous.frame != nullptr) {
			m_impl->frames.Release(previous.frame);
		}
		previous = layer;
	}
	m_impl->Present();
}

void Presenter::ClearLayer(int bus) {
	if (static_cast<size_t>(bus) >= m_impl->layers.size()) {
		return;
	}
	Common::LockGuard lock(m_impl->present_mutex);
	auto& layer = m_impl->layers[bus];
	if (layer.frame != nullptr) {
		m_impl->frames.Release(layer.frame);
		layer = {};
	}
}

void Presenter::Impl::Present() {
	KYTY_PROFILER_FUNCTION();

	const auto overlay_visual = GetSystemOverlayVisualState();
	// Keep black/blank flips in the current color space; only real main-bus frames switch it.
	const auto* main_frame = layers[0].frame;
	const bool hdr_changed = main_frame != nullptr && !main_frame->blank &&
	                         swapchain.SetHdr(main_frame->hdr);
	// Some window systems keep presenting an old swapchain after a resize.
	if (hdr_changed || swapchain.NeedsResize()) {
		RecoverSwapchain(Swapchain::Status::Recreate);
	}
	for (uint32_t attempt = 0; attempt < 2; attempt++) {
		auto status = swapchain.AcquireNextImage(present_scheduler);
		if (status != Swapchain::Status::Success) {
			RecoverSwapchain(status);
			continue;
		}
		// present_mutex owns the prepared frames and this scheduler. Only queue submission
		// is shared with rendering, and it has its own queue_mutex.
		auto&      command = present_scheduler.BeginCommand();
		const bool draw_system_overlay = (overlay_visual.active || PerformanceOverlayEnabled()) &&
		                                 swapchain.PrepareSystemOverlay();
		swapchain.RecordPresentCommands(command, layers[0].frame, layers[1],
		                                draw_system_overlay);
		const auto tick = swapchain.Submit(present_scheduler);
		for (const auto& layer: layers) {
			if (layer.frame != nullptr) {
				layer.frame->present_tick = tick;
			}
		}
		status = swapchain.Present();
		if (status != Swapchain::Status::Success) {
			RecoverSwapchain(status);
			continue;
		}

		presented_overlay_revision.store(overlay_visual.revision, std::memory_order_release);
		window.loop.presented_frames.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	LOGF("Vulkan presentation retry exhausted; dropping frame\n");
	Common::DebugCounters::Add(Common::DebugCounters::Counter::DroppedPresents);
}

void Presenter::Discard(Frame& frame) {
	m_impl->frames.Release(&frame);
}

WindowContext::WindowContext() = default;

} // namespace Libs::Graphics
