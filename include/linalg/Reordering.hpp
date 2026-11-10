#pragma once

// Bandwidth/profile reducing orderings of sparse matrix graphs.
//
// cuthill_mckee() / reverse_cuthill_mckee():
//   Cuthill & McKee (1969): breadth-first search from a starting node,
//   visiting the unnumbered neighbours of each node in order of increasing
//   degree. George (1971) observed that reversing the order (RCM) gives an
//   envelope (profile) that is never larger, usually much smaller (proved by
//   Liu & Sherman 1976), with the same bandwidth.
//
//   Starting nodes: one per connected component, a pseudo-peripheral node
//   found with the algorithm of George & Liu (1979) (repeated rooted level
//   structures, restarting from a minimum-degree node of the last level
//   while the eccentricity grows). Components are processed in the order of
//   their lowest-numbered node.
//
//   The graph is that of A + A^T without self-loops, so structurally
//   non-symmetric patterns are handled too. Ties (equal degrees) are broken
//   by the original index, which makes the ordering deterministic.
//
// The result is a Permutation (new_to_old, see Permutation.hpp) to be used
// with SymmetricPermutation: B = P A P^T has a reduced bandwidth/profile.

#include "linalg/Permutation.hpp"
#include "linalg/SparsityPattern.hpp"

#include <string_view>
#include <vector>

namespace hydrama::linalg {

Permutation cuthill_mckee(const SparsityPattern& pattern);
Permutation reverse_cuthill_mckee(const SparsityPattern& pattern);

// Pseudo-peripheral node of the connected component containing `start`
// (George & Liu 1979). Exposed for testing.
int pseudo_peripheral_node(const SparsityPattern& pattern, int start);

// Named orderings for configuration: "none" (identity) and "rcm".
const std::vector<std::string_view>& ordering_names();
bool is_ordering(std::string_view name);
// Throws std::invalid_argument for unknown names.
Permutation compute_ordering(std::string_view name, const SparsityPattern& pattern);

} // namespace hydrama::linalg
