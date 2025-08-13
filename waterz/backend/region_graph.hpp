#pragma once

#include "types.hpp"

#include <cstddef>
#include <iostream>
#include <map>

/**
 * Extract the region graph from a segmentation. Edges are annotated with the 
 * maximum affinity between the regions.
 *
 * @param aff [in]
 *              The affinity graph to read the affinities from.
 * @param seg [in]
 *              The segmentation.
 * @param max_segid [in]
 *              The highest ID in the segmentation.
 * @param statisticsProvider [in]
 *              A statistics provider to update on-the-fly.
 * @param region_graph [out]
 *              A reference to a region graph to store the result.
 */
template<typename AG, typename V, typename StatisticsProviderType>
inline
void
get_region_graph(
		const AG& aff,
		const V& seg,
		std::size_t max_segid,
		StatisticsProviderType& statisticsProvider,
		RegionGraph<typename V::element>& rg) {

	typedef typename AG::element F;
	typedef typename V::element ID;
	typedef RegionGraph<ID> RegionGraphType;
	typedef typename RegionGraphType::EdgeIdType EdgeIdType;

	std::ptrdiff_t zdim = aff.shape()[1];
	std::ptrdiff_t ydim = aff.shape()[2];
	std::ptrdiff_t xdim = aff.shape()[3];

	// list of affinities between pairs of regions
	std::vector<std::map<ID, std::vector<F>>> affinities(max_segid+1);

	EdgeIdType e;
	std::size_t p[3];
	for (p[0] = 0; p[0] < zdim; ++p[0])
		for (p[1] = 0; p[1] < ydim; ++p[1])
			for (p[2] = 0; p[2] < xdim; ++p[2]) {

				ID id1 = seg[p[0]][p[1]][p[2]];
				statisticsProvider.addVoxel(id1, p[2], p[1], p[0]);

				for (int d = 0; d < 3; d++) {

					if (p[d] == 0)
						continue;

					ID id2 = seg[p[0]-(d==0)][p[1]-(d==1)][p[2]-(d==2)];

					if (id1 != id2) {

						auto mm = std::minmax(id1, id2);
						affinities[mm.first][mm.second].push_back(aff[d][p[0]][p[1]][p[2]]);
					}
				}
			}

	for (ID id1 = 1; id1 <= max_segid; ++id1) {
		for (const auto& p: affinities[id1]) {

			// p.first is ID
			// p.second is list of affiliated edges
			EdgeIdType e = rg.addEdge(id1, p.first);
			statisticsProvider.notifyNewEdge(e);

			for (F affinity : p.second)
				statisticsProvider.addAffinity(e, affinity);
        }
    }

	std::cout << "Region graph number of edges: " << rg.edges().size() << std::endl;
}

template<typename AG, typename V, typename StatisticsProviderType>
inline
void
get_region_graph_xy(
		const AG& aff,
		const V& seg,
		std::size_t max_segid,
		StatisticsProviderType& statisticsProvider,
		RegionGraph<typename V::element>& rg) {

	typedef typename AG::element F;
	typedef typename V::element ID;
	typedef RegionGraph<ID> RegionGraphType;
	typedef typename RegionGraphType::EdgeIdType EdgeIdType;

	std::ptrdiff_t zdim = aff.shape()[1];
	std::ptrdiff_t ydim = aff.shape()[2];
	std::ptrdiff_t xdim = aff.shape()[3];

	// list of affinities between pairs of regions
	std::vector<std::map<ID, std::vector<F>>> affinities(max_segid+1);

	EdgeIdType e;
	std::size_t p[3];
	for (p[0] = 0; p[0] < zdim; ++p[0])
		for (p[1] = 0; p[1] < ydim; ++p[1])
			for (p[2] = 0; p[2] < xdim; ++p[2]) {

				ID id1 = seg[p[0]][p[1]][p[2]];
				statisticsProvider.addVoxel(id1, p[2], p[1], p[0]);

				for (int d = 1; d < 3; d++) {

					if (p[d] == 0)
						continue;

					ID id2 = seg[p[0]-(d==0)][p[1]-(d==1)][p[2]-(d==2)];

					if (id1 != id2) {

						auto mm = std::minmax(id1, id2);
						affinities[mm.first][mm.second].push_back(aff[d][p[0]][p[1]][p[2]]);
					}
				}
			}

	for (ID id1 = 1; id1 <= max_segid; ++id1) {
		for (const auto& p: affinities[id1]) {

			// p.first is ID
			// p.second is list of affiliated edges
			EdgeIdType e = rg.addEdge(id1, p.first);
			statisticsProvider.notifyNewEdge(e);

			for (F affinity : p.second)
				statisticsProvider.addAffinity(e, affinity);
        }
    }

	std::cout << "Region graph number of edges: " << rg.edges().size() << std::endl;
}

template<typename AG, typename V, typename StatisticsProviderType>
inline
void
get_region_graph_z(
		const AG& aff,
		const V& seg,
		std::size_t max_segid,
		StatisticsProviderType& statisticsProvider,
		RegionGraph<typename V::element>& rg) {

	typedef typename AG::element F;
	typedef typename V::element ID;
	typedef RegionGraph<ID> RegionGraphType;
	typedef typename RegionGraphType::EdgeIdType EdgeIdType;

	std::ptrdiff_t zdim = aff.shape()[1];
	std::ptrdiff_t ydim = aff.shape()[2];
	std::ptrdiff_t xdim = aff.shape()[3];

	// list of affinities between pairs of regions
	std::vector<std::map<ID, std::vector<F>>> affinities(max_segid+1);

	EdgeIdType e;
	std::size_t p[3];
    for (p[0] = 1; p[0] < zdim; ++p[0]){
        // start for z=1
		for (p[1] = 0; p[1] < ydim; ++p[1]){
			for (p[2] = 0; p[2] < xdim; ++p[2]) {

				ID id1 = seg[p[0]][p[1]][p[2]];
				statisticsProvider.addVoxel(id1, p[2], p[1], p[0]);
                ID id2 = seg[p[0]-1][p[1]][p[2]];
				statisticsProvider.addVoxel(id2, p[2], p[1], p[0]-1);

                if (id1 != id2) {
                    auto mm = std::minmax(id1, id2);
                    affinities[mm.first][mm.second].push_back(aff[0][p[0]][p[1]][p[2]]);
                }
			}
		}
    }

	for (ID id1 = 1; id1 <= max_segid; ++id1) {
		for (const auto& p: affinities[id1]) {

			// p.first is ID
			// p.second is list of affiliated edges
			EdgeIdType e = rg.addEdge(id1, p.first);
			statisticsProvider.notifyNewEdge(e);

			for (F affinity : p.second)
				statisticsProvider.addAffinity(e, affinity);
        }
    }

	std::cout << "Region graph number of edges: " << rg.edges().size() << std::endl;
}

// Trait to extract scalar type T from volume_ref_ptr<T>
template <typename T>
struct get_scalar_type_from_volume_ref_ptr;

template <typename T>
struct get_scalar_type_from_volume_ref_ptr<std::shared_ptr<boost::multi_array_ref<T, 3>>> {
    using type = T;
};

template<typename AG, typename V, typename StatisticsProviderType>
inline
void
get_region_graph_from_array(
        size_t    num_edge,
		const AG& rg_score,
		const V&  rg_id1,
		const V&  rg_id2,
		StatisticsProviderType& statisticsProvider,
		RegionGraph<typename V::element>& rg) {

	typedef typename AG::element F;
	typedef typename V::element ID;
	typedef RegionGraph<ID> RegionGraphType;
	typedef typename RegionGraphType::EdgeIdType EdgeIdType;

	EdgeIdType e;

	for (ID i = 0; i < num_edge; ++i) {
			// p.first is ID
			// p.second is list of affiliated edges
			EdgeIdType e = rg.addEdge(rg_id1[i], rg_id2[i]);
			statisticsProvider.notifyNewEdge(e);
    }

	std::cout << "Region graph number of edges: " << rg.edges().size() << std::endl;
}
