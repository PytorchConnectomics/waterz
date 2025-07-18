#ifndef C_SHARED_H
#define C_SHARED_H

#include <iostream>

typedef uint64_t SegID;
typedef uint32_t GtID;
//typedef uint8_t AffValue;
//typedef uint8_t ScoreValue;
typedef float AffValue;
typedef float ScoreValue;


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


struct WaterzState {
	int     context;
	Metrics metrics;
};

#endif
