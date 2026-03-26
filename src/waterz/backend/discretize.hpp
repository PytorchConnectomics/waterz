#ifndef WATERZ_DISCRETIZE_H__
#define WATERZ_DISCRETIZE_H__

#include <cstdint>
#include <algorithm>

// --- discretize: value → bin index ---

template <typename To, typename From, typename LevelsType>
inline To discretize(From value, LevelsType levels) {
	return std::min((To)(value*levels), (To)(levels-1));
}

// uint8_t input specialization: value IS the bin (identity for 256 bins).
template <typename To, typename LevelsType>
inline To discretize(uint8_t value, LevelsType levels) {
	if (levels == 256) return (To)value;
	return std::min((To)((unsigned)value * levels / 256), (To)(levels-1));
}

// --- undiscretize: bin index → value ---
// Partially specialized on output type To via struct dispatch.

namespace detail {

template <typename To>
struct Undiscretizer {
	template <typename From, typename LevelsType>
	static inline To run(From value, LevelsType levels) {
		return ((To)value + 0.5) / levels;
	}
};

// uint8_t output: bin IS the value (identity for 256 bins).
template <>
struct Undiscretizer<uint8_t> {
	template <typename From, typename LevelsType>
	static inline uint8_t run(From value, LevelsType levels) {
		if (levels == 256) return (uint8_t)value;
		return (uint8_t)((unsigned)value * 256 / levels);
	}
};

} // namespace detail

template <typename To, typename From, typename LevelsType>
inline To undiscretize(From value, LevelsType levels) {
	return detail::Undiscretizer<To>::run(value, levels);
}

#endif // WATERZ_DISCRETIZE_H__
