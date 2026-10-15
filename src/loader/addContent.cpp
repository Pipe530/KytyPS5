#include "loader/addContent.h"

#include "common/file.h"
#include "common/logging/log.h"
#include "common/singleton.h"
#include "common/stringUtils.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <nlohmann/json.hpp>
#include <regex>
#include <system_error>

namespace Loader {

namespace {

struct AddContentRegistry {
	std::vector<AddContentEntry> entries;
};

std::string DeriveEntitlementLabel(const std::string& content_id) {
	// A PS4/PS5 content id looks like "<dist>-<title_id>_<svc>-<label>".
	// The entitlement label used by AppContent/NpEntitlementAccess is the
	// final segment after the last '-'.
	const auto dash = content_id.rfind('-');
	if (dash == std::string::npos || dash + 1 >= content_id.size()) {
		return {};
	}

	auto label = content_id.substr(dash + 1);
	if (label.size() > 16) {
		LOGF("AddContent: entitlement label too long (truncated): %s\n", label.c_str());
		label.resize(16);
	}
	return label;
}

bool IsValidContentId(const std::string& id) {
	// 36 chars: <2 letters><4 digits>-<4 letters><5 digits>_<2 digits>-<16 chars, zero padded>
	static const std::regex re("^[A-Z]{2}[0-9]{4}-[A-Z]{4}[0-9]{5}_[0-9]{2}-[A-Z0-9]{16}$");
	return id.size() == 36 && std::regex_match(id, re);
}

void AddOrUpdateEntry(std::map<std::string, AddContentEntry>& by_label, std::string content_id,
                     std::filesystem::path mount_dir) {
	if (!IsValidContentId(content_id)) {
		LOGF("AddContent: invalid content_id, ignored: %s\n", content_id.c_str());
		return;
	}
	auto label = DeriveEntitlementLabel(content_id);
	if (label.empty()) {
		LOGF("AddContent: can't derive entitlement label from content_id: %s\n",
		     content_id.c_str());
		return;
	}

	auto it = by_label.find(label);
	if (it == by_label.end()) {
		by_label.emplace(label, AddContentEntry {std::move(content_id), label, std::move(mount_dir)});
	} else if (it->second.mount_dir.empty() && !mount_dir.empty()) {
		// Prefer an entry that points at real data over a bare unlock-only ini.
		it->second.mount_dir = std::move(mount_dir);
	}
}

void ParseParamJson(const std::filesystem::path& file_name,
                    std::map<std::string, AddContentEntry>& by_label) {
	Common::File f;
	f.Open(file_name, Common::File::Mode::Read);
	if (f.IsInvalid()) {
		return;
	}

	const auto buf = f.ReadWholeBuffer();
	f.Close();

	if (buf.empty()) {
		return;
	}

	const auto* json_begin = reinterpret_cast<const char*>(buf.data());
	const auto* json_end   = json_begin + buf.size();
	auto        param_json = nlohmann::json::parse(json_begin, json_end, nullptr, false);
	if (param_json.is_discarded() || !param_json.is_object()) {
		return;
	}

	const auto it = param_json.find("contentId");
	if (it == param_json.end() || !it->is_string()) {
		return;
	}

	// sce_sys/param.json -> content root is two levels up.
	auto mount_dir = file_name.parent_path();
	if (mount_dir.filename() == "sce_sys") {
		mount_dir = mount_dir.parent_path();
	}

	AddOrUpdateEntry(by_label, it->get<std::string>(), mount_dir);
	LOGF("AddContent: found dumped package: %s (%s)\n",
	     it->get_ref<const std::string&>().c_str(), Common::PathToString(mount_dir).c_str());
}

void ScanDirectory(const std::filesystem::path& dir,
                   std::map<std::string, AddContentEntry>& by_label) {
	std::error_code ec;
	std::filesystem::recursive_directory_iterator it(
	    dir, std::filesystem::directory_options::skip_permission_denied, ec);
	const std::filesystem::recursive_directory_iterator end;

	for (; !ec && it != end; it.increment(ec)) {
		const auto& entry = *it;

		std::error_code file_ec;
		if (!entry.is_regular_file(file_ec) || file_ec) {
			continue;
		}

		const auto& path = entry.path();
		const auto  name  = Common::PathToString(path.filename());

		if (Common::EqualNoCase(name, "param.json")) {
			ParseParamJson(path, by_label);
		}
	}
}

} // namespace

void AddContentLoad(const std::filesystem::path& dir, const std::string& title_id) {
	auto* reg = Common::Singleton<AddContentRegistry>::Instance();
	reg->entries.clear();

	if (dir.empty() || !Common::File::IsDirectoryExisting(dir)) {
		LOGF("AddContent: directory not found, skipping: %s\n",
		     Common::PathToString(dir).c_str());
		return;
	}

	LOGF("AddContent: scanning %s (title_id=%s)\n", Common::PathToString(dir).c_str(),
	     title_id.c_str());

	std::map<std::string, AddContentEntry> by_label;
	ScanDirectory(dir, by_label);

	reg->entries.reserve(by_label.size());
	for (auto& [label, entry]: by_label) {
		if (!title_id.empty() && entry.content_id.find(title_id) == std::string::npos) {
			LOGF("AddContent: skipping %s (belongs to another title, current=%s)\n",
			     entry.content_id.c_str(), title_id.c_str());
			continue;
		}
		LOGF("AddContent: registered %s -> label=%s, mount=%s\n", entry.content_id.c_str(),
		     entry.entitlement_label.c_str(),
		     entry.mount_dir.empty() ? "(none)" : Common::PathToString(entry.mount_dir).c_str());
		reg->entries.push_back(std::move(entry));
	}
}

void AddContentClear() {
	auto* reg = Common::Singleton<AddContentRegistry>::Instance();
	reg->entries.clear();
}

const std::vector<AddContentEntry>& AddContentGetAll() {
	auto* reg = Common::Singleton<AddContentRegistry>::Instance();
	return reg->entries;
}

const AddContentEntry* AddContentFind(const char* label, size_t label_size) {
	if (label == nullptr || label_size == 0) {
		return nullptr;
	}

	const auto len = strnlen(label, label_size);

	auto* reg = Common::Singleton<AddContentRegistry>::Instance();
	for (const auto& entry: reg->entries) {
		if (entry.entitlement_label.size() == len &&
		    std::memcmp(entry.entitlement_label.data(), label, len) == 0) {
			return &entry;
		}
	}

	return nullptr;
}

} // namespace Loader
