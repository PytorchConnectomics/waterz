#ifndef WATERZ_DISCRETIZE_H__
#define WATERZ_DISCRETIZE_H__

#include <cstdint>
#include <algorithm>

template <typename To, typename From, typename LevelsType>
inline To discretize(From value, LevelsType levels) {
	return std::min((To)(value*levels), (To)(levels-1));
}

// Specialization for uint8_t input: value IS the bin index (identity).
// This is the key optimization for uint8 affinity mode — no multiplication,
// no float conversion.
template <typename To, typename LevelsType>
inline To discretize(uint8_t value, LevelsType levels) {
	if (levels == 256) return (To)value;
	// For non-256 bins, scale down
	return std::min((To)((unsigned)value * levels / 256), (To)(levels-1));
}

template <typename To, typename From, typename LevelsType>
inline To undiscretize(From value, LevelsType levels) {
	return ((To)value + 0.5)/levels;
}

#endif // WATERZ_DISCRETIZE_H__
