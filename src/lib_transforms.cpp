/*
 * Copyright (c) 2011, Marc Lebrun <marc.lebrun@cmla.ens-cachan.fr>
 * Modified  (c) 2019, Thibaud Ehret <ehret.thibaud@gmail.com>
 * All rights reserved.
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later
 * version. You should have received a copy of this license along
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @file lib_transforms.cpp
 * @brief 1D and 2D wavelet transforms
 *
 * @author Marc Lebrun <marc.lebrun@cmla.ens-cachan.fr>
 **/

#include "lib_transforms.h"
#include <math.h>

#include <numeric>

#define MATLAB

using namespace std;

/**
 * @brief Compute a full 2D Bior 1.5 spline wavelet (normalized)
 *
 * @param input: vector on which the transform will be applied;
 * @param output: will contain the result;
 * @param N: size of the 2D patch (N x N) on which the 2D transform
 *           is applied. Must be a power of 2;
 * @param d_i: for convenience. Shift for input to access to the patch;
 * @param r_i: for convenience. input(i, j) = input[d_i + i * r_i + j];
 * @param d_o: for convenience. Shift for output;
 * @param lpd: low frequencies coefficients for the forward Bior 1.5;
 * @param hpd: high frequencies coefficients for the forward Bior 1.5.
 *
 * @return none.
 **/
#ifndef MATLAB
void bior_2d_forward(Video<float> const& input,
                     vector<float>& output,
                     const uint64_t N,
                     const uint64_t x,
                     const uint64_t y,
                     const uint64_t t,
                     const uint64_t c,
                     const uint64_t d_o,
                     vector<float> const& lpd,
                     vector<float> const& hpd)
{
    //! Initializing output
    for(uint64_t i = 0; i < N; i++)
        for(uint64_t j = 0; j < N; j++)
            output[i * N + j + d_o] = input(x + j, y + i, t, c);

    const uint64_t iter_max = log2(N);
    uint64_t N_1 = N;
    uint64_t N_2 = N / 2;
    const uint64_t S_1 = lpd.size();
    const uint64_t S_2 = S_1 / 2 - 1;

    for(uint64_t iter = 0; iter < iter_max; iter++)
    {
        //! Periodic extension index initialization
        vector<float> tmp(N_1 + 2 * S_2);
        vector<uint64_t> ind_per(N_1 + 2 * S_2);
        per_ext_ind(ind_per, N_1, S_2);

        //! Implementing row filtering
        for(uint64_t i = 0; i < N_1; i++)
        {
            //! Periodic extension of the signal in row
            for(uint64_t j = 0; j < tmp.size(); j++)
                tmp[j] = output[d_o + i * N + ind_per[j]];

            //! Low and High frequencies filtering
            for(uint64_t j = 0; j < N_2; j++)
            {
                float v_l = 0.0f, v_h = 0.0f;
                for(uint64_t k = 0; k < S_1; k++)
                {
                    v_l += tmp[k + j * 2] * lpd[k];
                    v_h += tmp[k + j * 2] * hpd[k];
                }
                output[d_o + i * N + j] = v_l;
                output[d_o + i * N + j + N_2] = v_h;
            }
        }

        //! Implementing column filtering
        for(uint64_t j = 0; j < N_1; j++)
        {
            //! Periodic extension of the signal in column
            for(uint64_t i = 0; i < tmp.size(); i++)
                tmp[i] = output[d_o + j + ind_per[i] * N];

            //! Low and High frequencies filtering
            for(uint64_t i = 0; i < N_2; i++)
            {
                float v_l = 0.0f, v_h = 0.0f;
                for(uint64_t k = 0; k < S_1; k++)
                {
                    v_l += tmp[k + i * 2] * lpd[k];
                    v_h += tmp[k + i * 2] * hpd[k];
                }
                output[d_o + j + i * N] = v_l;
                output[d_o + j + (i + N_2) * N] = v_h;
            }
        }

        //! Sizes update
        N_1 /= 2;
        N_2 /= 2;
    }
}
#else
void bior_2d_forward(Video<float> const& input,
                     vector<float>& output,
                     const uint64_t N,
                     const uint64_t x,
                     const uint64_t y,
                     const uint64_t t,
                     const uint64_t c,
                     const uint64_t d_o,
                     vector<float> const& lpd,
                     vector<float> const& hpd)
{
    assert(N == 8);

    // Bior1.5 basis extracted from MATLAB
    std::vector<float> bior_basis(8 * 8);
    bior_basis[0 + 8 * 0] = 0.353553390593274;
    bior_basis[1 + 8 * 0] = 0.353553390593274;
    bior_basis[2 + 8 * 0] = 0.353553390593274;
    bior_basis[3 + 8 * 0] = 0.353553390593274;
    bior_basis[4 + 8 * 0] = 0.353553390593274;
    bior_basis[5 + 8 * 0] = 0.353553390593274;
    bior_basis[6 + 8 * 0] = 0.353553390593274;
    bior_basis[7 + 8 * 0] = 0.353553390593274;
    bior_basis[0 + 8 * 1] = 0.219417649252501;
    bior_basis[1 + 8 * 1] = 0.449283757993216;
    bior_basis[2 + 8 * 1] = 0.449283757993216;
    bior_basis[3 + 8 * 1] = 0.219417649252501;
    bior_basis[4 + 8 * 1] = -0.219417649252501;
    bior_basis[5 + 8 * 1] = -0.449283757993216;
    bior_basis[6 + 8 * 1] = -0.449283757993216;
    bior_basis[7 + 8 * 1] = -0.219417649252501;
    bior_basis[0 + 8 * 2] = 0.569359398342846;
    bior_basis[1 + 8 * 2] = 0.402347308162278;
    bior_basis[2 + 8 * 2] = -0.402347308162278;
    bior_basis[3 + 8 * 2] = -0.569359398342846;
    bior_basis[4 + 8 * 2] = -0.083506045090284;
    bior_basis[5 + 8 * 2] = 0.083506045090284;
    bior_basis[6 + 8 * 2] = -0.083506045090284;
    bior_basis[7 + 8 * 2] = 0.083506045090284;
    bior_basis[0 + 8 * 3] = -0.083506045090284;
    bior_basis[1 + 8 * 3] = 0.083506045090284;
    bior_basis[2 + 8 * 3] = -0.083506045090284;
    bior_basis[3 + 8 * 3] = 0.083506045090284;
    bior_basis[4 + 8 * 3] = 0.569359398342846;
    bior_basis[5 + 8 * 3] = 0.402347308162278;
    bior_basis[6 + 8 * 3] = -0.402347308162278;
    bior_basis[7 + 8 * 3] = -0.569359398342846;
    bior_basis[0 + 8 * 4] = 0.707106781186547;
    bior_basis[1 + 8 * 4] = -0.707106781186547;
    bior_basis[2 + 8 * 4] = 0.0;
    bior_basis[3 + 8 * 4] = 0.0;
    bior_basis[4 + 8 * 4] = 0.0;
    bior_basis[5 + 8 * 4] = 0.0;
    bior_basis[6 + 8 * 4] = 0.0;
    bior_basis[7 + 8 * 4] = 0.0;
    bior_basis[0 + 8 * 5] = 0.0;
    bior_basis[1 + 8 * 5] = 0.0;
    bior_basis[2 + 8 * 5] = 0.707106781186547;
    bior_basis[3 + 8 * 5] = -0.707106781186547;
    bior_basis[4 + 8 * 5] = 0.0;
    bior_basis[5 + 8 * 5] = 0.0;
    bior_basis[6 + 8 * 5] = 0.0;
    bior_basis[7 + 8 * 5] = 0.0;
    bior_basis[0 + 8 * 6] = 0.0;
    bior_basis[1 + 8 * 6] = 0.0;
    bior_basis[2 + 8 * 6] = 0.0;
    bior_basis[3 + 8 * 6] = 0.0;
    bior_basis[4 + 8 * 6] = 0.707106781186547;
    bior_basis[5 + 8 * 6] = -0.707106781186547;
    bior_basis[6 + 8 * 6] = 0.0;
    bior_basis[7 + 8 * 6] = 0.0;
    bior_basis[0 + 8 * 7] = 0.0;
    bior_basis[1 + 8 * 7] = 0.0;
    bior_basis[2 + 8 * 7] = 0.0;
    bior_basis[3 + 8 * 7] = 0.0;
    bior_basis[4 + 8 * 7] = 0.0;
    bior_basis[5 + 8 * 7] = 0.0;
    bior_basis[6 + 8 * 7] = 0.707106781186547;
    bior_basis[7 + 8 * 7] = -0.707106781186547;

    //! Initializing
    std::vector<float> temp(8 * 8);
    std::vector<float> temp2(8 * 8);

    for(uint64_t i = 0; i < N; i++)
        for(uint64_t j = 0; j < N; j++)
            temp[i * N + j] = input(x + j, y + i, t, c);

    // First pass
    for(uint64_t i = 0; i < N; i++)
        for(uint64_t j = 0; j < N; j++)
        {
            float mult = 0.;
            for(uint64_t k = 0; k < N; k++)
                mult += temp[i * N + k] * bior_basis[k + j * N];
            temp2[i * N + j] = mult;
        }

    // Second pass
    for(uint64_t i = 0; i < N; i++)
        for(uint64_t j = 0; j < N; j++)
        {
            float mult = 0.;
            for(uint64_t k = 0; k < N; k++)
                mult += temp2[k * N + i] * bior_basis[k + j * N];
            temp[i * N + j] = mult;
        }

    for(uint64_t i = 0; i < N; i++)
        for(uint64_t j = 0; j < N; j++)
            output[i * N + j + d_o] = temp[i * N + j];
}
#endif

/**
 * @brief Compute a full 2D Bior 1.5 spline wavelet inverse (normalized)
 *
 * @param signal: vector on which the transform will be applied; It
 *                will contain the result at the end;
 * @param N: size of the 2D patch (N x N) on which the 2D transform
 *           is applied. Must be a power of 2;
 * @param d_s: for convenience. Shift for signal to access to the patch;
 * @param lpr: low frequencies coefficients for the inverse Bior 1.5;
 * @param hpr: high frequencies coefficients for the inverse Bior 1.5.
 *
 * @return none.
 **/
#ifndef MATLAB
void bior_2d_inverse(vector<float>& signal,
                     const uint64_t N,
                     const uint64_t d_s,
                     vector<float> const& lpr,
                     vector<float> const& hpr)
{
    //! Initialization
    const uint64_t iter_max = log2(N);
    uint64_t N_1 = 2;
    uint64_t N_2 = 1;
    const uint64_t S_1 = lpr.size();
    const uint64_t S_2 = S_1 / 2 - 1;

    for(uint64_t iter = 0; iter < iter_max; iter++)
    {

        vector<float> tmp(N_1 + S_2 * N_1);
        vector<uint64_t> ind_per(N_1 + 2 * S_2 * N_2);
        per_ext_ind(ind_per, N_1, S_2 * N_2);

        //! Implementing column filtering
        for(uint64_t j = 0; j < N_1; j++)
        {
            //! Periodic extension of the signal in column
            for(uint64_t i = 0; i < tmp.size(); i++)
                tmp[i] = signal[d_s + j + ind_per[i] * N];

            //! Low and High frequencies filtering
            for(uint64_t i = 0; i < N_2; i++)
            {
                float v_l = 0.0f, v_h = 0.0f;
                for(uint64_t k = 0; k < S_1; k++)
                {
                    v_l += lpr[k] * tmp[k * N_2 + i];
                    v_h += hpr[k] * tmp[k * N_2 + i];
                }

                signal[d_s + i * 2 * N + j] = v_h;
                signal[d_s + (i * 2 + 1) * N + j] = v_l;
            }
        }

        //! Implementing row filtering
        for(uint64_t i = 0; i < N_1; i++)
        {
            //! Periodic extension of the signal in row
            for(uint64_t j = 0; j < tmp.size(); j++)
                tmp[j] = signal[d_s + i * N + ind_per[j]];

            //! Low and High frequencies filtering
            for(uint64_t j = 0; j < N_2; j++)
            {
                float v_l = 0.0f, v_h = 0.0f;
                for(uint64_t k = 0; k < S_1; k++)
                {
                    v_l += lpr[k] * tmp[k * N_2 + j];
                    v_h += hpr[k] * tmp[k * N_2 + j];
                }

                signal[d_s + i * N + j * 2] = v_h;
                signal[d_s + i * N + j * 2 + 1] = v_l;
            }
        }

        //! Sizes update
        N_1 *= 2;
        N_2 *= 2;
    }
}
#else
void bior_2d_inverse(vector<float>& signal,
                     const uint64_t N,
                     const uint64_t d_s,
                     vector<float> const& lpr,
                     vector<float> const& hpr)
{
    assert(N == 8);

    // Bior1.5 basis extracted from MATLAB
    std::vector<float> bior_basis(8 * 8);
    bior_basis[0 + 8 * 0] = 0.353553390593274;
    bior_basis[1 + 8 * 0] = 0.353553390593274;
    bior_basis[2 + 8 * 0] = 0.5;
    bior_basis[3 + 8 * 0] = 0.0;
    bior_basis[4 + 8 * 0] = 0.707106781186547;
    bior_basis[5 + 8 * 0] = -0.121533978016438;
    bior_basis[6 + 8 * 0] = 0.0;
    bior_basis[7 + 8 * 0] = 0.121533978016438;
    bior_basis[0 + 8 * 1] = 0.353553390593274;
    bior_basis[1 + 8 * 1] = 0.353553390593274;
    bior_basis[2 + 8 * 1] = 0.5;
    bior_basis[3 + 8 * 1] = -0.0;
    bior_basis[4 + 8 * 1] = -0.707106781186547;
    bior_basis[5 + 8 * 1] = -0.121533978016438;
    bior_basis[6 + 8 * 1] = 0.0;
    bior_basis[7 + 8 * 1] = 0.121533978016438;
    bior_basis[0 + 8 * 2] = 0.353553390593274;
    bior_basis[1 + 8 * 2] = 0.353553390593274;
    bior_basis[2 + 8 * 2] = -0.5;
    bior_basis[3 + 8 * 2] = -0.0;
    bior_basis[4 + 8 * 2] = 0.121533978016438;
    bior_basis[5 + 8 * 2] = 0.707106781186548;
    bior_basis[6 + 8 * 2] = -0.121533978016438;
    bior_basis[7 + 8 * 2] = 0.0;
    bior_basis[0 + 8 * 3] = 0.353553390593274;
    bior_basis[1 + 8 * 3] = 0.353553390593274;
    bior_basis[2 + 8 * 3] = -0.5;
    bior_basis[3 + 8 * 3] = 0.0;
    bior_basis[4 + 8 * 3] = 0.121533978016438;
    bior_basis[5 + 8 * 3] = -0.707106781186548;
    bior_basis[6 + 8 * 3] = -0.121533978016438;
    bior_basis[7 + 8 * 3] = 0.0;
    bior_basis[0 + 8 * 4] = 0.353553390593274;
    bior_basis[1 + 8 * 4] = -0.353553390593274;
    bior_basis[2 + 8 * 4] = 0.0;
    bior_basis[3 + 8 * 4] = 0.5;
    bior_basis[4 + 8 * 4] = 0.0;
    bior_basis[5 + 8 * 4] = 0.121533978016438;
    bior_basis[6 + 8 * 4] = 0.707106781186547;
    bior_basis[7 + 8 * 4] = -0.121533978016438;
    bior_basis[0 + 8 * 5] = 0.353553390593274;
    bior_basis[1 + 8 * 5] = -0.353553390593274;
    bior_basis[2 + 8 * 5] = 0.0;
    bior_basis[3 + 8 * 5] = 0.5;
    bior_basis[4 + 8 * 5] = 0.0;
    bior_basis[5 + 8 * 5] = 0.121533978016438;
    bior_basis[6 + 8 * 5] = -0.707106781186547;
    bior_basis[7 + 8 * 5] = -0.121533978016438;
    bior_basis[0 + 8 * 6] = 0.353553390593274;
    bior_basis[1 + 8 * 6] = -0.353553390593274;
    bior_basis[2 + 8 * 6] = 0.0;
    bior_basis[3 + 8 * 6] = -0.5;
    bior_basis[4 + 8 * 6] = -0.121533978016438;
    bior_basis[5 + 8 * 6] = 0.0;
    bior_basis[6 + 8 * 6] = 0.121533978016438;
    bior_basis[7 + 8 * 6] = 0.707106781186547;
    bior_basis[0 + 8 * 7] = 0.353553390593274;
    bior_basis[1 + 8 * 7] = -0.353553390593274;
    bior_basis[2 + 8 * 7] = 0.0;
    bior_basis[3 + 8 * 7] = -0.5;
    bior_basis[4 + 8 * 7] = -0.121533978016438;
    bior_basis[5 + 8 * 7] = 0.0;
    bior_basis[6 + 8 * 7] = 0.121533978016438;
    bior_basis[7 + 8 * 7] = -0.707106781186547;

    //! Initializing
    std::vector<float> temp(8 * 8);
    std::vector<float> temp2(8 * 8);

    for(uint64_t i = 0; i < N; i++)
        for(uint64_t j = 0; j < N; j++)
            temp[i * N + j] = signal[d_s + i + j * N];

    // First pass
    for(uint64_t i = 0; i < N; i++)
        for(uint64_t j = 0; j < N; j++)
        {
            float mult = 0.;
            for(uint64_t k = 0; k < N; k++)
                mult += temp[i * N + k] * bior_basis[k + j * N];
            temp2[i * N + j] = mult;
        }

    // Second pass
    for(uint64_t i = 0; i < N; i++)
        for(uint64_t j = 0; j < N; j++)
        {
            float mult = 0.;
            for(uint64_t k = 0; k < N; k++)
                mult += temp2[k * N + i] * bior_basis[k + j * N];
            temp[i * N + j] = mult;
        }

    for(uint64_t i = 0; i < N; i++)
        for(uint64_t j = 0; j < N; j++)
            signal[j * N + i + d_s] = temp[i * N + j];
}
#endif

/**
 * @brief Initialize forward and backward low and high filter
 *        for a Bior1.5 spline wavelet.
 *
 * @param lp1: low frequencies forward filter;
 * @param hp1: high frequencies forward filter;
 * @param lp2: low frequencies backward filter;
 * @param hp2: high frequencies backward filter.
 **/
void bior15_coef(vector<float>& lp1, vector<float>& hp1, vector<float>& lp2, vector<float>& hp2)
{
    const float coef_norm = 1.f / (sqrtf(2.f) * 128.f);
    const float sqrt2_inv = 1.f / sqrtf(2.f);

    lp1.resize(10);
    lp1[0] = 3.f;
    lp1[1] = -3.f;
    lp1[2] = -22.f;
    lp1[3] = 22.f;
    lp1[4] = 128.f;
    lp1[5] = 128.f;
    lp1[6] = 22.f;
    lp1[7] = -22.f;
    lp1[8] = -3.f;
    lp1[9] = 3.f;

    hp1.resize(10);
    hp1[0] = 0.f;
    hp1[1] = 0.f;
    hp1[2] = 0.f;
    hp1[3] = 0.f;
    hp1[4] = -sqrt2_inv;
    hp1[5] = sqrt2_inv;
    hp1[6] = 0.f;
    hp1[7] = 0.f;
    hp1[8] = 0.f;
    hp1[9] = 0.f;

    lp2.resize(10);
    lp2[0] = 0.f;
    lp2[1] = 0.f;
    lp2[2] = 0.f;
    lp2[3] = 0.f;
    lp2[4] = sqrt2_inv;
    lp2[5] = sqrt2_inv;
    lp2[6] = 0.f;
    lp2[7] = 0.f;
    lp2[8] = 0.f;
    lp2[9] = 0.f;

    hp2.resize(10);
    hp2[0] = 3.f;
    hp2[1] = 3.f;
    hp2[2] = -22.f;
    hp2[3] = -22.f;
    hp2[4] = 128.f;
    hp2[5] = -128.f;
    hp2[6] = 22.f;
    hp2[7] = 22.f;
    hp2[8] = -3.f;
    hp2[9] = -3.f;

    for(uint64_t k = 0; k < 10; k++)
    {
        lp1[k] *= coef_norm;
        hp2[k] *= coef_norm;
    }
}

/**
 * @brief Apply Welsh-Hadamard transform on vec (non normalized !!)
 *
 * @param vec: vector on which a Hadamard transform will be applied.
 *        It will contain the transform at the end;
 * @param tmp: must have the same size as vec. Used for convenience;
 * @param N, d: the Hadamard transform will be applied on vec[d] -> vec[d + N].
 *        N must be a power of 2!!!!
 *
 * @return None.
 **/
void hadamard_transform(vector<float>& vec, vector<float>& tmp, const uint64_t N, const uint64_t D)
{
    if(N == 1)
        return;
    else if(N == 2)
    {
        const float a = vec[D + 0];
        const float b = vec[D + 1];
        vec[D + 0] = a + b;
        vec[D + 1] = a - b;
    } else
    {
        const uint64_t n = N / 2;
        for(uint64_t k = 0; k < n; k++)
        {
            const float a = vec[D + 2 * k];
            const float b = vec[D + 2 * k + 1];
            vec[D + k] = a + b;
            tmp[k] = a - b;
        }
        for(uint64_t k = 0; k < n; k++)
            vec[D + n + k] = tmp[k];
        hadamard_transform(vec, tmp, n, D);
        hadamard_transform(vec, tmp, n, D + n);
    }
}

/**
 * @brief Apply Haar transform on vec (normalized !!)
 *
 * @param vec: vector on which a Haar transform will be applied.
 *        It will contain the transform at the end;
 * @param tmp: must have the same size as vec. Used for convenience;
 * @param N, d: the Hadamard transform will be applied on vec[d] -> vec[d + N].
 *        N must be a power of 2!!!!
 *
 * @return None.
 **/
void haar_forward(vector<float>& vec, vector<float>& tmp, const uint64_t N, const uint64_t D)
{
    for(int i = 0; i < tmp.size(); ++i)
        tmp[i] = 0.f;

    int k = N;
    while(1 < k)
    {
        k = k / 2;
        for(int i = 0; i < k; i++)
        {
            tmp[i] = (vec[D + 2 * i] + vec[D + 2 * i + 1]) * SQRT2_INV;
            tmp[i + k] = (vec[D + 2 * i] - vec[D + 2 * i + 1]) * SQRT2_INV;
        }
        for(int i = 0; i < k * 2; i++)
        {
            vec[D + i] = tmp[i];
        }
    }
}

/**
 * @brief Apply the inverse Haar transform on vec (normalized !!)
 *
 * @param vec: vector on which a inverse Haar transform will be applied.
 *        It will contain the transform at the end;
 * @param tmp: must have the same size as vec. Used for convenience;
 * @param N, d: the Hadamard transform will be applied on vec[d] -> vec[d + N].
 *        N must be a power of 2!!!!
 *
 * @return None.
 **/
void haar_inverse(vector<float>& vec, vector<float>& tmp, const uint64_t N, const uint64_t D)
{
    for(int i = 0; i < tmp.size(); ++i)
        tmp[i] = 0.f;

    int k = 1;
    while(k * 2 <= N)
    {
        for(int i = 0; i < k; ++i)
        {
            tmp[2 * i] = (vec[D + i] + vec[D + i + k]) * SQRT2_INV;
            tmp[2 * i + 1] = (vec[D + i] - vec[D + i + k]) * SQRT2_INV;
        }
        for(int i = 0; i < k * 2; i++)
        {
            vec[D + i] = tmp[i];
        }
        k = k * 2;
    }
}

/**
 * @brief Obtain the ceil of log_2(N)
 *
 * @param N: in the case N = 2^n, return n.
 *
 * @return n;
 **/
uint64_t log2(const uint64_t N)
{
    uint64_t k = 1;
    uint64_t n = 0;
    while(k < N)
    {
        k *= 2;
        n++;
    }
    return n;
}

/**
 * @brief Obtain index for periodic extension.
 *
 * @param ind_per: will contain index. Its size must be N + 2 * L;
 * @param N: size of the original signal;
 * @param L: size of boundaries to add on each side of the signal.
 *
 * @return none.
 **/
void per_ext_ind(vector<uint64_t>& ind_per, const uint64_t N, const uint64_t L)
{
    for(uint64_t k = 0; k < N; k++)
        ind_per[k + L] = k;

    int ind1 = (N - L);
    while(ind1 < 0)
        ind1 += N;
    uint64_t ind2 = 0;
    uint64_t k = 0;
    while(k < L)
    {
        ind_per[k] = (uint64_t)ind1;
        ind_per[k + L + N] = ind2;
        ind1 = ((uint64_t)ind1 < N - 1 ? (uint64_t)ind1 + 1 : 0);
        ind2 = (ind2 < N - 1 ? ind2 + 1 : 0);
        k++;
    }
}
