#include <memory>

#include <iostream>
#include <algorithm>
#include <vector>

#include "frontend_agglomerate.h"
#include "evaluate.hpp"
#include "backend/MergeFunctions.hpp"
#include "backend/basic_watershed.hpp"
#include "backend/region_graph.hpp"

std::map<int, WaterzContext*> WaterzContext::_contexts;
int WaterzContext::_nextId = 0;

WaterzState
initialize(
		std::size_t     width,
		std::size_t     height,
		std::size_t     depth,
		const AffValue* affinity_data,
		SegID*          segmentation_data,
		const GtID*     ground_truth_data,
		AffValue        affThresholdLow,
		AffValue        affThresholdHigh,
		bool            findFragments) {

	std::size_t num_voxels = width*height*depth;

	// wrap affinities (no copy)
	affinity_graph_ref<AffValue> affinities(
			affinity_data,
			boost::extents[3][width][height][depth]
	);

	// wrap segmentation array (no copy)
	volume_ref_ptr<SegID> segmentation(
			new volume_ref<SegID>(
					segmentation_data,
					boost::extents[width][height][depth]
			)
	);

	counts_t<std::size_t> sizes;

	if (findFragments) {

		std::cout << "performing initial watershed segmentation..." << std::endl;

		watershed(affinities, affThresholdLow, affThresholdHigh, *segmentation, sizes);

	} else {

		std::cout << "counting regions and sizes..." << std::endl;

		std::size_t maxId = *std::max_element(segmentation_data, segmentation_data + num_voxels);
		sizes.resize(maxId + 1);
		for (std::size_t i = 0; i < num_voxels; i++)
			sizes[segmentation_data[i]]++;
	}

	std::size_t numNodes = sizes.size();
	std::cout << "creating region graph for " << numNodes << " nodes" << std::endl;

	std::shared_ptr<RegionGraphType> regionGraph(
			new RegionGraphType(numNodes)
	);

	std::cout << "creating statistics provider" << std::endl;
	std::shared_ptr<StatisticsProviderType> statisticsProvider(
			new StatisticsProviderType(*regionGraph)
	);

	std::cout << "extracting region graph..." << std::endl;

	get_region_graph(
			affinities,
			*segmentation,
			numNodes - 1,
			*statisticsProvider,
			*regionGraph);

	std::shared_ptr<ScoringFunctionType> scoringFunction(
			new ScoringFunctionType(*regionGraph, *statisticsProvider)
	);

	std::shared_ptr<RegionMergingType> regionMerging(
			new RegionMergingType(*regionGraph)
	);

	WaterzContext* context = WaterzContext::createNew();
	context->regionGraph        = regionGraph;
	context->regionMerging      = regionMerging;
	context->scoringFunction    = scoringFunction;
	context->statisticsProvider = statisticsProvider;
	context->segmentation       = segmentation;

	WaterzState initial_state;
	initial_state.context = context->id;

	if (ground_truth_data != NULL) {

		// wrap ground-truth (no copy)
		volume_const_ref_ptr<GtID> groundtruth(
				new volume_const_ref<GtID>(
						ground_truth_data,
						boost::extents[width][height][depth]
				)
		);

		context->groundtruth = groundtruth;
	}

	return initial_state;
}

std::vector<Merge>
mergeUntil(
		WaterzState& state,
		ScoreValue   threshold) {

	WaterzContext* context = WaterzContext::get(state.context);

	std::cout << "merging until threshold " << (double)threshold << std::endl;

	std::vector<Merge>  mergeHistory;
	MergeHistoryVisitor mergeHistoryVisitor(mergeHistory);

	std::size_t merged = context->regionMerging->mergeUntil(
			*context->scoringFunction,
			*context->statisticsProvider,
			threshold,
			mergeHistoryVisitor);

	if (merged) {

		std::cout << "extracting segmentation" << std::endl;

		context->regionMerging->extractSegmentation(*context->segmentation);
	}

	if (context->groundtruth) {

		std::cout << "evaluating current segmentation against ground-truth" << std::endl;

		auto m = compare_volumes(*context->groundtruth, *context->segmentation);

		state.metrics.rand_split = std::get<0>(m);
		state.metrics.rand_merge = std::get<1>(m);
		state.metrics.voi_split  = std::get<2>(m);
		state.metrics.voi_merge  = std::get<3>(m);
	}

	return mergeHistory;
}

std::vector<ScoredEdge>
getRegionGraph(WaterzState& state, bool rescore) {

	WaterzContext* context = WaterzContext::get(state.context);
	std::shared_ptr<RegionMergingType> regionMerging = context->regionMerging;
	std::shared_ptr<ScoringFunctionType> scoringFunction = context->scoringFunction;

	std::cout << "getRegionGraph: rescore=" << rescore
	          << ", numEdges=" << context->regionGraph->numEdges()
	          << std::endl;

	auto result = regionMerging->extractRegionGraph<ScoredEdge>(*scoringFunction, rescore);

	std::cout << "getRegionGraph: returned " << result.size() << " edges" << std::endl;

	return result;
}

std::vector<ScoredEdge>
buildRegionGraphOnly(
		std::size_t     width,
		std::size_t     height,
		std::size_t     depth,
		const AffValue* affinity_data,
		SegID*          segmentation_data) {

	std::size_t num_voxels = width*height*depth;

	// wrap arrays (no copy)
	affinity_graph_ref<AffValue> affinities(
			affinity_data,
			boost::extents[3][width][height][depth]
	);
	volume_ref<SegID> segmentation_vol(
			segmentation_data,
			boost::extents[width][height][depth]
	);

	// count sizes
	std::size_t maxId = *std::max_element(segmentation_data, segmentation_data + num_voxels);
	counts_t<std::size_t> sizes(maxId + 1, 0);
	for (std::size_t i = 0; i < num_voxels; i++)
		sizes[segmentation_data[i]]++;

	std::size_t numNodes = sizes.size();

	// build region graph + statistics (the essential work)
	RegionGraphType regionGraph(numNodes);
	StatisticsProviderType statisticsProvider(regionGraph);

	get_region_graph(
			affinities,
			segmentation_vol,
			numNodes - 1,
			statisticsProvider,
			regionGraph);

	// score edges directly — NO RegionMerging, NO priority queue
	ScoringFunctionType scoringFunction(regionGraph, statisticsProvider);

	std::vector<ScoredEdge> edges;
	edges.reserve(regionGraph.edges().size());

	for (RegionGraphType::EdgeIdType e = 0; e < regionGraph.numEdges(); e++) {
		ScoreValue score = scoringFunction(e);
		edges.push_back(ScoredEdge(
				regionGraph.edge(e).u,
				regionGraph.edge(e).v,
				score));
	}

	return edges;
}

std::vector<RichScoredEdge>
buildRegionGraphRich(
		std::size_t     width,
		std::size_t     height,
		std::size_t     depth,
		const AffValue* affinity_data,
		SegID*          segmentation_data) {

	std::size_t num_voxels = width*height*depth;

	// wrap arrays (no copy)
	affinity_graph_ref<AffValue> affinities(
			affinity_data,
			boost::extents[3][width][height][depth]
	);
	volume_ref<SegID> segmentation_vol(
			segmentation_data,
			boost::extents[width][height][depth]
	);

	// count sizes
	std::size_t maxId = *std::max_element(segmentation_data, segmentation_data + num_voxels);
	counts_t<std::size_t> sizes(maxId + 1, 0);
	for (std::size_t i = 0; i < num_voxels; i++)
		sizes[segmentation_data[i]]++;

	std::size_t numNodes = sizes.size();

	// build region graph + statistics with edge counts
	RegionGraphType regionGraph(numNodes);
	StatisticsProviderType statisticsProvider(regionGraph);
	std::vector<uint64_t> edge_counts;

	get_region_graph(
			affinities,
			segmentation_vol,
			numNodes - 1,
			statisticsProvider,
			regionGraph,
			edge_counts);

	// score edges directly — NO RegionMerging, NO priority queue
	ScoringFunctionType scoringFunction(regionGraph, statisticsProvider);

	std::vector<RichScoredEdge> edges;
	edges.reserve(regionGraph.edges().size());

	for (RegionGraphType::EdgeIdType e = 0; e < regionGraph.numEdges(); e++) {
		ScoreValue score = scoringFunction(e);
		edges.push_back(RichScoredEdge(
				regionGraph.edge(e).u,
				regionGraph.edge(e).v,
				score,
				edge_counts[e]));
	}

	return edges;
}

void
free(WaterzState& state) {

	WaterzContext::free(state.context);
}
