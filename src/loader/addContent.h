#ifndef EMULATOR_INCLUDE_EMULATOR_LOADER_ADDCONTENT_H_
#define EMULATOR_INCLUDE_EMULATOR_LOADER_ADDCONTENT_H_

#include "common/common.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace Loader {

// Describes one DLC / additional-content entry discovered on the host.
//
// Scene "fake license" DLC dumps only ship a small ini file (see
// AddContentLoad()) that proves ownership; the actual asset data already
// lives inside the base game thanks to a title update, so `mount_dir` stays
// empty for those. Properly dumped AddCont packages that ship their own
// sce_sys/param.sfo (or param.json) alongside real data are mountable and
// get `mount_dir` populated with their data folder.
struct AddContentEntry {
	std::string           content_id;
	std::string           entitlement_label; /* last 16 chars of content_id */
	std::filesystem::path mount_dir;         /* empty if there is no separate data to mount */
};

// Scans `dir` (and its sub-directories) for additional content entries and
// registers them for the current title. Safe to call with a non-existing
// directory (it's simply a no-op). `title_id` is used only to build the
// fallback placeholder mount directories for unlock-only entries.
void AddContentLoad(const std::filesystem::path& dir, const std::string& title_id);

void AddContentClear();

const std::vector<AddContentEntry>& AddContentGetAll();

// Looks up an entry by its (up to 16 byte) entitlement label. `label` may be
// shorter than 16 bytes; comparison stops at the first NUL.
const AddContentEntry* AddContentFind(const char* label, size_t label_size);

} // namespace Loader

#endif /* EMULATOR_INCLUDE_EMULATOR_LOADER_ADDCONTENT_H_ */
