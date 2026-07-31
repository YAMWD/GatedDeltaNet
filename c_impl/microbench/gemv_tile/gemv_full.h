#ifndef GEMV_FULL_H
#define GEMV_FULL_H

#include <stdint.h>

#define GEMV_FULL_CHANNELS 32
#define GEMV_FULL_CLUSTERS 8
#define GEMV_FULL_CHANNELS_PER_CLUSTER 4
#define GEMV_TILE_PARTIAL 8
#define GEMV_TILE_LANES 16
#define GEMV_TILE_IN_DIM_MAX 5632

struct Pack16 {
    float data[GEMV_TILE_LANES];
};

#endif
