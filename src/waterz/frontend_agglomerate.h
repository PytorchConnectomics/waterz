#ifndef C_FRONTEND_H
#define C_FRONTEND_H

#include <vector>

#include "backend/IterativeRegionMerging.hpp"
#include "backend/MergeFunctions.hpp"
#include "backend/Operators.hpp"
#include "backend/types.hpp"
#include "backend/BinQueue.hpp"
#include "backend/PriorityQueue.hpp"
#include "backend/HistogramQuantileProvider.hpp"
#include "backend/VectorQuantileProvider.hpp"
#include "evaluate.hpp"

typedef uint32_t GtID;

// AffValue, ScoreValue, SegID, ScoringFunctionType, QueueType
// are defined in the JIT-generated headers below.
#include <SegType.h>
#include <AffType.h>

typedef RegionGraph<SegID> RegionGraphType;
#include <ScoringFunction.h>
#include <Queue.h>

typedef typename ScoringFunctionType::StatisticsProviderType StatisticsProviderType;
typedef IterativeRegionMerging<SegID, ScoreValue, QueueType> RegionMergingType;

struct Metrics {

	double voi_split;
	double voi_merge;
	double rand_split;
	double rand_merge;
};

struct Merge {

	SegID a;
	SegID b;
	SegID c;
	ScoreValue score;
};

struct ScoredEdge {

	ScoredEdge(SegID u_, SegID v_, ScoreValue score_) :
		u(u_),
		v(v_),
		score(score_) {}

	SegID u;
	SegID v;
	ScoreValue score;
};

struct RichScoredEdge {

	RichScoredEdge(SegID u_, SegID v_, ScoreValue score_, uint64_t contact_area_) :
		u(u_),
		v(v_),
		score(score_),
		contact_area(contact_area_) {}

	SegID u;
	SegID v;
	ScoreValue score;
	uint64_t contact_area;
};

struct WaterzState {

	int     context;
	Metrics metrics;
};

class WaterzContext {

public:

	static WaterzContext* createNew() {

		WaterzContext* context = new WaterzContext();
		context->id = _nextId;
		_nextId++;
		_contexts.insert(std::make_pair(context->id, context));

		return context;
	}

	static WaterzContext* get(int id) {

		if (!_contexts.count(id))
			return NULL;

		return _contexts.at(id);
	}

	static void free(int id) {

		WaterzContext* context = get(id);

		if (context) {

			_contexts.erase(id);
			delete context;
		}
	}

	int id;

	std::shared_ptr<RegionGraphType> regionGraph;
	std::shared_ptr<RegionMergingType> regionMerging;
	std::shared_ptr<ScoringFunctionType> scoringFunction;
	std::shared_ptr<StatisticsProviderType> statisticsProvider;
	volume_ref_ptr<SegID> segmentation;
	volume_const_ref_ptr<GtID> groundtruth;

private:

	WaterzContext() {}

	~WaterzContext() {}

	static std::map<int, WaterzContext*> _contexts;
	static int _nextId;
};

class RegionMergingVisitor {

public:

	void onPop(RegionGraphType::EdgeIdType e, ScoreValue score) {}

	void onDeletedEdgeFound(RegionGraphType::EdgeIdType e) {}

	void onStaleEdgeFound(RegionGraphType::EdgeIdType e, ScoreValue oldScore, ScoreValue newScore) {}

	void onMerge(SegID a, SegID b, SegID c, ScoreValue score) {}
};

class MergeHistoryVisitor : public RegionMergingVisitor {

public:

	MergeHistoryVisitor(std::vector<Merge>& history) : _history(history) {}

	void onMerge(SegID a, SegID b, SegID c, ScoreValue score) {

		_history.push_back({a, b, c, score});
	}

private:

	std::vector<Merge>& _history;
};

WaterzState initialize(
		size_t          width,
		size_t          height,
		size_t          depth,
		const AffValue* affinity_data,
		SegID*          segmentation_data,
		const GtID*     groundtruth_data = NULL,
		AffValue        affThresholdLow  = 0.0001,
		AffValue        affThresholdHigh = 0.9999,
		bool            findFragments = true);

std::vector<Merge> mergeUntil(
		WaterzState& state,
		ScoreValue   threshold);

std::vector<ScoredEdge> getRegionGraph(WaterzState& state, bool rescore = true);

/**
 * Build region graph and extract scored edges without creating
 * RegionMerging or priority queue.  Much faster than
 * initialize() + mergeUntil(0) + getRegionGraph() when only the
 * scored edge list is needed.
 */
std::vector<ScoredEdge> buildRegionGraphOnly(
		std::size_t     width,
		std::size_t     height,
		std::size_t     depth,
		const AffValue* affinity_data,
		SegID*          segmentation_data);

/**
 * Like buildRegionGraphOnly but also returns per-edge contact area
 * (number of affinity samples contributing to the edge score).
 */
std::vector<RichScoredEdge> buildRegionGraphRich(
		std::size_t     width,
		std::size_t     height,
		std::size_t     depth,
		const AffValue* affinity_data,
		SegID*          segmentation_data);

void free(WaterzState& state);

#endif
