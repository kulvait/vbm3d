/*
 * Copyright (c) 2013, Marc Lebrun <marc.lebrun.ik@gmail.com>
 * All rights reserved.
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later
 * version. You should have received a copy of this license along
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef UTILITIES_H_INCLUDED
#define UTILITIES_H_INCLUDED

#include <vector>
#include "LibImages.h"

/**
 * @brief Convenient function to use the sort function provided by the vector library.
 **/
bool comparaisonFirst(
	const std::pair<float, uint64_t> &i_pair1
,	const std::pair<float, uint64_t> &i_pair2
);

bool comparaisonInverseFirst(
	const std::pair<float, uint64_t> &i_pair1
,	const std::pair<float, uint64_t> &i_pair2
);

/**
 * @brief Clip a value between min and max
 *
 * @param i_value: value to clip;
 * @param i_min: minimum value;
 * @param i_max: maximum value.
 *
 * @return value clipped between [min, max].
 **/
float clip(
	const float i_value
,	const float i_min
,	const float i_max
);

/**
 * @brief Obtain and substract the baricenter of io_group3d.
 *
 * @param io_group3d(p_rows x p_cols) : data to center;
 * @param o_baricenter(p_cols): will contain the baricenter of io_group3d;
 * @param p_rows, p_cols: size of io_group3d.
 *
 * @return none.
 **/
void centerData(
	std::vector<float> &io_group3d
,	std::vector<float> &o_baricenter
,	const uint64_t p_rows
,	const uint64_t p_cols
);

/**
 * @brief Compute the average standard deviation of a set of patches.
 *
 * @param i_Set(p_sP, p_nSimP): set of patches;
 * @param p_sP : size of a patch;
 * @param p_nSimP: number of patches in the set;
 * @param p_nChannels: number of channels of the image.
 *
 * @return the average standard deviation of the set
 **/
float computeStdDeviation(
	std::vector<float> const& i_Set
,	const uint64_t p_sP
,	const uint64_t p_nSimP
,	const uint64_t p_nChannels
);

/**
 * @brief Determine a and b such that : n = a * b, with a and b as greatest as possible
 *
 * @param i_n : number to decompose;
 * @param o_a : will contain a;
 * @param o_b : will contain b.
 *
 * @return none.
 **/
void determineFactor(
    const uint64_t i_n
,   uint64_t &o_a
,   uint64_t &o_b
);

/**
 * @brief Write PSNR and RMSE in a .txt for both basic and denoised images.
 *
 * @param p_pathName: path name of the file;
 * @param p_sigma: value of the noise;
 * @param p_psnr: value of the PSNR of the denoised image;
 * @param p_rmse: value of the RMSE of the denoised image;
 * @param p_truncateFile: if true, erase the file when open it. Otherwise
 *        write at the end of the file;
 * @param p_app: in order to specify the image.
 *
 * @return EXIT_FAILURE if the file can't be opened.
 **/
int writingMeasures(
    const char* p_pathName
,   const float p_sigma
,   const float p_psnr
,   const float p_rmse
,   const float p_grps
,   const float p_time
,   const float p_cons
,   const bool  p_truncateFile
,   const char* p_app
);

//! Check if a number is a power of 2
bool power_of_2(
    const uint64_t n
);

//! Look for the closest power of 2 number
int closest_power_of_2(
    const uint64_t n
);

//! Estimate sigma on each channel according to the choice of the color_space
int estimate_sigma(
    const float sigma
,   std::vector<float> &sigma_table
,   const uint64_t chnls
,   const uint64_t color_space
);

//! Initialize a 2D fftwf_plan with some parameters
void allocate_plan_2d(
    fftwf_plan* plan
,   const uint64_t N
,   const fftwf_r2r_kind kind
,   const uint64_t nb
);

//! Initialize a 1D fftwf_plan with some parameters
void allocate_plan_1d(
    fftwf_plan* plan
,   const uint64_t N
,   const fftwf_r2r_kind kind
,   const uint64_t nb
);

//! Initialize a set of indices
void ind_initialize(
    std::vector<uint64_t> &ind_set
,   const uint64_t beginning
,   const uint64_t end
,   const uint64_t step
);

void ind_initialize2(
    std::vector<uint64_t> &ind_set
,   const uint64_t max_size
,   const uint64_t N
,   const uint64_t step
);

//! For convenience
uint64_t ind_size(
    const uint64_t beginning
,   const uint64_t end
,   const uint64_t step
);


#endif // UTILITIES_H_INCLUDED
