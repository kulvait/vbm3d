/*
 * Copyright (c) 2018, Thibaud Ehret <ehret.thibaud@gmail.com>
 * All rights reserved.
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later
 * version. You should have received a copy of this license along
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef VBM3D_H_INCLUDED
#define VBM3D_H_INCLUDED

#include "Utilities/LibVideoT.hpp"
#include "parameters.h"
#include <fftw3.h>
#include <vector>

/** ------------------ **/
/** - Main functions - **/
/** ------------------ **/
//! Main function
int run_vbm3d(const float sigma,
              Video<float>& vid_noisy,
              Video<float>& fflow,
              Video<float>& bflow,
              Video<float>& vid_basic,
              Video<float>& vid_denoised,
              const Parameters& prms_1,
              const Parameters& prms_2,
              const bool color_space,
              volatile sig_atomic_t& sigterm_caught);

//! 1st step of VBM3D
void vbm3d_1st_step(const float sigma,
                    Video<float> const& vid_noisy,
                    Video<float>& fflow,
                    Video<float>& bflow,
                    const Parameters& prms,
                    fftwf_plan* plan_2d,
                    fftwf_plan* plan_2d_inv,
                    fftwf_plan* plan_1d,
                    fftwf_plan* plan_1d_inv,
                    VideoUtils::CropPosition* crop,
                    const bool color_space,
                    Video<float>& numberator,
                    Video<float>& denominator);

//! 2nd step of VBM3D
void vbm3d_2nd_step(const float sigma,
                    Video<float> const& vid_noisy,
                    Video<float> const& vid_basic,
                    Video<float>& fflow,
                    Video<float>& bflow,
                    const Parameters& prms,
                    fftwf_plan* plan_2d,
                    fftwf_plan* plan_2d_inv,
                    fftwf_plan* plan_1d,
                    fftwf_plan* plan_1d_inv,
                    VideoUtils::CropPosition* crop,
                    const bool color_space,
                    Video<float>& numberator,
                    Video<float>& denominator);

//! Process 2D dct of a group of patches
void dct_2d_process(std::vector<float>& DCT_table_2D,
                    Video<float> const& vid,
                    std::vector<uint64_t> const& patch_table,
                    fftwf_plan* plan,
                    std::vector<float> const& coef_norm,
                    Video<float>& fflow,
                    const Parameters& prms);

int computeSimilarPatches(std::vector<float>& distances,
                          std::vector<uint64_t>& indexes,
                          uint64_t idx,
                          const Video<float>& vid,
                          Video<float>& fflow,
                          Video<float>& bflow,
                          const Parameters& prms);

//! Process 2D bior1.5 transform of a group of patches
void bior_2d_process(std::vector<float>& bior_table_2D,
                     Video<float> const& vid,
                     std::vector<uint64_t> const& patch_table,
                     std::vector<float>& lpd,
                     std::vector<float>& hpd,
                     Video<float>& fflow,
                     const Parameters& prms);

void dct_2d_inv(std::vector<float>& group_3D_table,
                const uint64_t kHW,
                const uint64_t ktHW,
                const uint64_t N,
                std::vector<float> const& coef_norm_inv,
                fftwf_plan* plan);

void bior_2d_inv(std::vector<float>& group_3D_table,
                 const uint64_t kHW,
                 const uint64_t ktHW,
                 std::vector<float> const& lpr,
                 std::vector<float> const& hpr);

//! HT filtering using Welsh-Hadamard transform (do only
//! third dimension transform, Hard Thresholding
//! and inverse Hadamard transform)
void ht_filtering_hadamard(std::vector<float>& group_3D,
                           std::vector<float>& tmp,
                           const uint64_t nSx_r,
                           const uint64_t kHard,
                           const uint64_t ktHard,
                           const uint64_t chnls,
                           std::vector<float> const& sigma_table,
                           const float lambdaThr3D,
                           std::vector<float>& weight_table);

//! HT filtering using Haar transform (do only
//! third dimension transform, Hard Thresholding
//! and inverse Hadamard transform)
void ht_filtering_haar(std::vector<float>& group_3D,
                       std::vector<float>& tmp,
                       const uint64_t nSx_r,
                       const uint64_t kHard,
                       const uint64_t ktHard,
                       const uint64_t chnls,
                       std::vector<float> const& sigma_table,
                       const float lambdaThr3D,
                       std::vector<float>& weight_table);

//! Wiener filtering using Welsh-Hadamard transform
void wiener_filtering_hadamard(std::vector<float>& group_3D_img,
                               std::vector<float>& group_3D_est,
                               std::vector<float>& tmp,
                               const uint64_t nSx_r,
                               const uint64_t kWien,
                               const uint64_t ktWien,
                               const uint64_t chnls,
                               std::vector<float> const& sigma_table,
                               std::vector<float>& weight_table);

//! Wiener filtering using Haar transform
void wiener_filtering_haar(std::vector<float>& group_3D_img,
                           std::vector<float>& group_3D_est,
                           std::vector<float>& tmp,
                           const uint64_t nSx_r,
                           const uint64_t kWien,
                           const uint64_t ktWien,
                           const uint64_t chnls,
                           std::vector<float> const& sigma_table,
                           std::vector<float>& weight_table);

//! Apply a bior1.5 spline wavelet on a vector of size N x N.
void bior1_5_transform(std::vector<float> const& input,
                       std::vector<float>& output,
                       const uint64_t N,
                       std::vector<float> const& bior_table,
                       const uint64_t d_i,
                       const uint64_t d_o,
                       const uint64_t N_i,
                       const uint64_t N_o);

void temporal_transform(std::vector<float>& group_3D,
                        const uint64_t kHW,
                        const uint64_t ktHW,
                        const uint64_t chnls,
                        const uint64_t nSx_r,
                        const uint64_t N,
                        fftwf_plan* plan);

void temporal_inv_transform(std::vector<float>& group_3D,
                            const uint64_t kHW,
                            const uint64_t ktHW,
                            const uint64_t chnls,
                            const uint64_t nSx_r,
                            const uint64_t N,
                            fftwf_plan* plan);

/** ---------------------------------- **/
/** - Preprocessing / Postprocessing - **/
/** ---------------------------------- **/
//! Preprocess coefficients of the Kaiser window and normalization coef for the DCT
void preProcess(std::vector<float>& kaiserWindow,
                std::vector<float>& coef_norm,
                std::vector<float>& coef_norm_inv,
                const uint64_t kHW);

#endif // VBM3D_H_INCLUDED
