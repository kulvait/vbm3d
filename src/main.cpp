/*
 * Copyright (c) 2018, Thibaud Ehret <ehret.thibaud@gmail.com>
 * Copyright (c) 2018, Pablo Arias <pablo.arias@upf.edu>
 * All rights reserved.
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later
 * version. You should have received a copy of this license along
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <csignal>
#include <ctype.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <memory>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "Utilities/LibVideoT.hpp"
#include "Utilities/Utilities.h"
#include "Utilities/cmd_option.h"
#include "vbm3d.h"

#define DCT 0
#define BIOR 1
#define HADAMARD 2
#define HAAR 3
#define NONE -1

using namespace std;

volatile sig_atomic_t sigterm_caught = 0;

void signal_handler(int signal_num)
{
    if(signal_num == SIGTERM)
    {
        printf("Received SIGTERM\n");
        sigterm_caught = 1;
    } else
    {

        cout << "The interrupt signal is (" << signal_num << "). \n";

        // It terminates the  program
        exit(signal_num);
    }
}

// Signal handler
// https://stackoverflow.com/questions/8400530/how-can-i-tell-in-linux-which-process-sent-my-process-a-signal/8400532#8400532
static std::string get_process_name(pid_t pid)
{
    std::string name = "<unknown>";

    std::ifstream file("/proc/" + std::to_string(pid) + "/comm");
    if(file.is_open())
    {
        std::getline(file, name);
    }
    return name;
}

static void signal_handler_pid(int signal_num, siginfo_t* info, void* context)
{
    //    printf("Signal %s received from process %s with PID %d.\n", strsignal_numal(sign),
    //           get_process_name(info->si_pid).c_str(), info->si_pid);
    std::string SIGNALNAME;
    // char* SIGNALNAME = sigabbrev_np(signal_num);
    if(signal_num == SIGTERM)
    {
        SIGNALNAME = "SIGTERM";
        sigterm_caught = 1;
    } else if(signal_num == SIGINT)
    {
        SIGNALNAME = "SIGINT";
    } else
    {
        SIGNALNAME = strdup(sys_siglist[signal_num]);
    }
    printf("Signal %s received from process %s with PID %d.\n", SIGNALNAME.c_str(),
           get_process_name(info->si_pid).c_str(), info->si_pid);

    switch(signal_num)
    {
    case SIGINT:
        exit(signal_num);
        break;
    case SIGTERM:
        break;
    default:
        break;
    }
}

void initializeParameters_1(Parameters& prms,
                            const int k,
                            const int kt,
                            const int Nf,
                            const int Ns,
                            const int Npr,
                            const int Nb,
                            const int p,
                            const int N,
                            const int d,
                            const float tau,
                            const float lambda3D,
                            const unsigned T_2D,
                            const unsigned T_3D,
                            const float sigma)
{
    if(k < 0)
        prms.k = 8;
    else
        prms.k = k;

    if(kt < 0)
        prms.kt = 1;
    else
        prms.kt = kt;

    if(Nf < 0)
        prms.Nf = 4;
    else
        prms.Nf = Nf;
    if(Ns < 0)
        prms.Ns = 7;
    else
        prms.Ns = Ns;
    if(Npr < 0)
        prms.Npr = 5;
    else
        prms.Npr = Npr;
    if(Nb < 0)
        prms.Nb = 2;
    else
        prms.Nb = Nb;

    if(d < 0)
        prms.d = (7. * 7. * 255.) / (prms.k * prms.k * prms.kt);
    else
        prms.d = (d * d * 255.) / (prms.k * prms.k * prms.kt);

    if(p < 0)
        // In the original VBM3D article, the authors used a step of 6 for a patch of size 8 (so 3/4
        // of the size of the patch). Therefore, we generalize that ratio to all patch size as a
        // default step parameter.
        prms.p = prms.k / 4 * 3;
    else
        prms.p = p;

    if(N < 0)
        prms.N = 8;
    else
        prms.N = N;

    if(tau < 0)
        prms.tau = (sigma > 30) ? 4500 : 3000;
    else
        prms.tau = tau;

    if(lambda3D < 0)
        prms.lambda3D = 2.7f;
    else
        prms.lambda3D = lambda3D;

    if(T_2D == NONE)
    {
        if(prms.k == 8)
            prms.T_2D = BIOR;
        else
            prms.T_2D = DCT;
    } else
    {
        if(prms.k == 8)
            prms.T_2D = T_2D;
        else
            prms.T_2D = DCT;
    }

    if(T_3D == NONE)
        prms.T_3D = HAAR;
    else
        prms.T_3D = T_3D;
}

void initializeParameters_2(Parameters& prms,
                            const int k,
                            const int kt,
                            const int Nf,
                            const int Ns,
                            const int Npr,
                            const int Nb,
                            const int p,
                            const int N,
                            const int d,
                            const float tau,
                            const unsigned T_2D,
                            const unsigned T_3D,
                            const float sigma)
{
    if(k < 0)
        prms.k = (sigma > 30) ? 8 : 7;
    else
        prms.k = k;

    if(kt < 0)
        prms.kt = 1;
    else
        prms.kt = kt;

    if(Nf < 0)
        prms.Nf = 4;
    else
        prms.Nf = Nf;

    if(Ns < 0)
        prms.Ns = 7;
    else
        prms.Ns = Ns;
    if(Npr < 0)
        prms.Npr = 5;
    else
        prms.Npr = Npr;
    if(Nb < 0)
        prms.Nb = 2;
    else
        prms.Nb = Nb;

    if(d < 0)
        prms.d = (3. * 3. * 255.) / (prms.k * prms.k * prms.kt);
    else
        prms.d = (d * d * 255.) / (prms.k * prms.k * prms.kt);

    if(p < 0)
        // In the original VBM3D article, the authors used a step of 4 for a patch of size 8 (so 1/2
        // of the size of the patch). Therefore, we generalize that ratio to all patch size as a
        // default step parameter.
        prms.p = prms.k / 2;
    else
        prms.p = p;

    if(N < 0)
        prms.N = 8;
    else
        prms.N = N;

    if(tau < 0)
        prms.tau = (sigma > 30) ? 3000 : 1500;
    else
        prms.tau = tau;

    if(T_2D == NONE)
        prms.T_2D = DCT;
    else
    {
        if(prms.k == 8)
            prms.T_2D = T_2D;
        else
            prms.T_2D = DCT;
    }

    if(T_3D == NONE)
        prms.T_3D = HAAR;
    else
        prms.T_3D = T_3D;
}

void printParameters(const Parameters& prms, const string suffix)
{
    const char* s = suffix.c_str();
    if(!prms.k)
    {
        printf("%s step skipped\n", s);
        return;
    }

    printf("Parameters %s step:\n", s);
    printf("\tpatch size k x k x kt = %lux%lux%lu\n", prms.k, prms.k, prms.kt);
    printf("\tprocessing step p  = %lu\n", prms.p);
    printf("\tpred. search Nf    = %lu\n", prms.Nf);
    printf("\tpred. search Ns    = %lu\n", prms.Ns);
    printf("\tpred. search Npr   = %lu\n", prms.Npr);
    printf("\tpred. search Nb    = %lu\n", prms.Nb);
    printf("\tpred. search d     = %f\n", prms.d);
    printf("\tpred. search N     = %lu\n", prms.N);
    printf("\tpred. search tau   = %f\n", prms.tau);
    printf("\ttransform T_2D     = %lu\n", prms.T_2D);
    printf("\ttransform T_3D     = %lu\n", prms.T_3D);
    printf("\tthreshold lambda3D = %f\n", prms.lambda3D);
    printf("\tmotion comp.       = %s\n", prms.mc ? "true" : "false");
    return;
}

/**
 * @file   main.cpp
 * @brief  Main executable file.
 * @author Thibaud Ehret  <ehret.thibaud@gmail.com>
 */

static struct sigaction sigact;

int main(int argc, char** argv)
{ // C++ signal handling
  // signal(SIGTERM, signal_handler);
    // stdlib signal handling including PID
    // https://stackoverflow.com/questions/8400530/how-can-i-tell-in-linux-which-process-sent-my-process-a-signal/8400532#8400532
    sigact.sa_sigaction = signal_handler_pid;
    sigact.sa_flags = SA_SIGINFO;
    sigaction(SIGTERM, &sigact, NULL);
    sigaction(SIGINT, &sigact, NULL);

    // Handling signals
    //! Check if there is the right call for the algorithm
    using std::string;
    const string input_path = clo_option("-i", "", "< Input sequence");
    const string inbsc_path = clo_option("-b", "", "< Input basic sequence (replacing first pass)");
    const string noisy_path = clo_option("-nisy", "", "> Noisy sequence (only when -add is true)");
    const string final_path = clo_option("-deno", "deno_%03d.tiff", "> Denoised sequence");
    const string basic_path = clo_option("-bsic", "bsic_%03d.tiff", "> Basic denoised sequence");
    const string diff_path = clo_option(
        "-diff", "", "> Difference sequence between noisy and output (only when -add is true)");
    const string diffi_path
        = clo_option("-diffi", "", "> Difference sequence between input and output");
    const string meas_path
        = clo_option("-meas", "measure.txt", "> Text file with PSNR/RMSE (only when -add is true)");
    const string fflow_path = clo_option("-fflow", "", "< Forward optical flow ");
    const string bflow_path = clo_option("-bflow", "", "< Backward optical flow ");

    const unsigned firstFrame = clo_option("-f", 0, "< Index of the first frame");
    const unsigned lastFrame = clo_option("-l", 0, "< Index of the last frame");
    const unsigned frameStep = clo_option("-s", 1, "< Frame step");

    //! General parameters
    const float fSigma = clo_option("-sigma", 0.f, "< Noise standard deviation");
    const bool addnoise = (bool)clo_option("-add", true, "< Add noise");
    const bool verbose = (bool)clo_option("-verbose", true, "> Verbose output");

    //! VBM3D parameters
    const int kHard = clo_option("-kHard", NONE, "< Spatial size of the patch (first pass)");
    const int ktHard = clo_option("-ktHard", NONE, "< Temporal size of the patch (first pass)");
    const int NfHard = clo_option(
        "-NfHard", NONE, "< Number frames used before and after the reference (first pass)");
    const int NsHard = clo_option(
        "-NsHard", NONE, "< Size of the search region in the reference frame (first pass)");
    const int NprHard = clo_option("-NprHard", NONE,
                                   "< Size of the search region in the other frames (first pass)");
    const int NbHard
        = clo_option("-NbHard", NONE, "< Maximum number of neighbors per frame (first pass)");
    const int pHard = clo_option("-pHard", NONE, "< Step between each patch (first pass)");
    const int NHard = clo_option("-NHard", NONE, "< Maximum number of neighbors (first pass)");
    const int dHard = clo_option("-dHard", NONE, "< Bias toward center patches (first pass)");
    const float tauHard
        = clo_option("-tauHard", NONE, "< Distance threshold on neighbors (first pass)");
    const float lambda3D = clo_option("-lambda3d", NONE, "< Coefficient threhsold (first pass)");
    const unsigned T_2D_hard = (unsigned)clo_option(
        "-T2dh", NONE, "< 2D transform (first pass), choice is 1 (dct) or 2 (bior)");
    const unsigned T_3D_hard = (unsigned)clo_option(
        "-T3dh", NONE, "< 1D transform (first pass), choice is 3 (hadamard) or 4 (haar)");

    const int kWien = clo_option("-kWien", NONE, "< Spatial size of the patch (second pass)");
    const int ktWien = clo_option("-ktWien", NONE, "< Temporal size of the patch (second pass)");
    const int NfWien = clo_option(
        "-NfWien", NONE, "< Number of frames used before and after the reference (second pass)");
    const int NsWien = clo_option(
        "-NsWien", NONE, "< Size of the search region in the reference frame (second pass)");
    const int NprWien = clo_option("-NprWien", NONE,
                                   "< Size of the search region in the other frames (second pass)");
    const int NbWien
        = clo_option("-NbWien", NONE, "< Maximum number of neighbors per frame (second pass)");
    const int pWien = clo_option("-pWien", NONE, "< Step between each patch (second pass)");
    const int NWien = clo_option("-NWien", NONE, "< Maximum number of neighbors (second pass)");
    const int dWien = clo_option("-dWien", NONE, "< Bias toward center patches (second pass)");
    const float tauWien
        = clo_option("-tauWien", NONE, "< Distance threshold on neighbors (second pass)");
    const unsigned T_2D_wien = (unsigned)clo_option(
        "-T2dw", NONE, "< 2D transform (second pass), choice is 0 (dct) or 1 (bior)");
    const unsigned T_3D_wien = (unsigned)clo_option(
        "-T3dw", NONE, "< 1D transform (first pass), choice is 2 (hadamard) or 3 (haar)");

    const bool color_space
        = (bool)clo_option("-color", true, "< Transform the color space during the processing");
    const bool mc = (bool)clo_option("-mc", false, "< Motion compensation for 3d patches");

    //! Check inputs
    if(input_path == "")
        return fprintf(stderr,
                       "%s: no input images.\n"
                       "Try `%s --help' for more information.\n",
                       argv[0], argv[0]),
               EXIT_FAILURE;

    //! Check for invalid inputs
    if(T_2D_hard != NONE && T_2D_hard != DCT && T_2D_hard != BIOR)
        return fprintf(stderr,
                       "%s: unknown T_2d_hard."
                       "Try `%s --help' for the available choices.\n",
                       argv[0], argv[0]),
               EXIT_FAILURE;

    if(T_2D_wien != NONE && T_2D_wien != DCT && T_2D_wien != BIOR)
        return fprintf(stderr,
                       "%s: unknown T_2d_wien."
                       "Try `%s --help' for the available choices.\n",
                       argv[0], argv[0]),
               EXIT_FAILURE;

    if(T_3D_hard != NONE && T_3D_hard != HAAR && T_3D_hard != HADAMARD)
        return fprintf(stderr,
                       "%s: unknown T_3d_hard."
                       "Try `%s --help' for the available choices.\n",
                       argv[0], argv[0]),
               EXIT_FAILURE;

    if(T_3D_wien != NONE && T_3D_wien != HAAR && T_3D_wien != HADAMARD)
        return fprintf(stderr,
                       "%s: unknown T_3d_wien."
                       "Try `%s --help' for the available choices.\n",
                       argv[0], argv[0]),
               EXIT_FAILURE;

    //! Init parameters
    Parameters prms_1;
    Parameters prms_2;

    initializeParameters_1(prms_1, kHard, ktHard, NfHard, NsHard, NprHard, NbHard, pHard, NHard,
                           dHard, tauHard, lambda3D, T_2D_hard, T_3D_hard, fSigma);
    prms_1.mc = mc;
    initializeParameters_2(prms_2, kWien, ktWien, NfWien, NsWien, NprWien, NbWien, pWien, NWien,
                           dWien, tauWien, T_2D_wien, T_3D_wien, fSigma);
    prms_2.mc = mc;

    //! Print parameters
    if(verbose)
    {
        printParameters(prms_1, "hard thresholding");
        printParameters(prms_2, "Wiener filtering");
    }

    //! Declarations
    std::shared_ptr<Video<float>> vid, vid_noisy, vid_basic, vid_denoised, vid_diff;
    Video<float> fflow, bflow;

    // Compute proper dimensions
    char filename[1024];
    std::sprintf(filename, input_path.c_str(), firstFrame);
    ImageSize frameSize;
    std::vector<float> frame;
    if(loadImage(filename, frame, frameSize) == EXIT_FAILURE)
        throw std::runtime_error("Video<T>::loadVideo: loading of " + std::string(filename)
                                 + " failed");
    VideoSize sz(frameSize.width, frameSize.height, (lastFrame - firstFrame + 1) / frameStep,
                 frameSize.nChannels);

    // First allocation then expensive loading of data
    // This way we can prevent std::bac_alloc after data are loaded
    printf("Allocating fflow memory for video data...\n");
    if(fflow_path == "")
    {
        fflow.resize(sz.width, sz.height, sz.frames, 2);
    } else
    {
        fflow.loadFullFlow(fflow_path, firstFrame, lastFrame - 1, frameStep);
    }

    printf("Allocating bflow memory for video data...\n");
    if(bflow_path == "")
    {
        bflow.resize(sz.width, sz.height, sz.frames, 2);
    } else
    {
        bflow.loadFullFlow(bflow_path, firstFrame + 1, lastFrame, frameStep);
    }
    vid = std::make_shared<Video<float>>(sz, 0.0f);
    vid_basic = std::make_shared<Video<float>>(sz, 0.0f);
    vid_denoised = std::make_shared<Video<float>>(sz, 0.0f);
    //! Load video
    printf("Loading video data...\n");
    vid->loadVideo(input_path, firstFrame, lastFrame, frameStep);
    if(kHard == 0)
    {
        printf("kHard = %d, vid_basic will be loaded from file\n", kHard);
        vid_basic->loadVideo(inbsc_path, firstFrame, lastFrame, frameStep);
    } else
    {
        printf("kHard = %d, vid_basis won't be loaded from file\n", kHard);
    }
    // Check that all sizes are consistent
    if(fflow.sz.width != bflow.sz.width || fflow.sz.width != vid->sz.width
       || fflow.sz.height != bflow.sz.height || fflow.sz.height != vid->sz.height)
        return fprintf(stderr,
                       "%s: incompatible flows and/or frame sizes.\n"
                       "Try `%s --help' for more information.\n",
                       argv[0], argv[0]),
               EXIT_FAILURE;

    //! Add noise
    if(addnoise)
    {
        vid_noisy = std::make_shared<Video<float>>(sz, 0.0f);
        VideoUtils::addNoise(*vid, *vid_noisy, fSigma, verbose);
    } else
    {
        printf("Noisy input video with sigma = %f\n", fSigma);
        vid_noisy = vid;
    }

    //! Denoising
    if(run_vbm3d(fSigma, *vid_noisy, fflow, bflow, *vid_basic, *vid_denoised, prms_1, prms_2,
                 color_space, sigterm_caught)
       != EXIT_SUCCESS)
        return EXIT_FAILURE;

    //! Compute PSNR and RMSE
    if(addnoise)
    {
        double psnr, rmse;
        double psnr_basic, rmse_basic;
        VideoUtils::computePSNR(*vid, *vid_basic, psnr_basic, rmse_basic);
        VideoUtils::computePSNR(*vid, *vid_denoised, psnr, rmse);

        if(verbose)
        {
            printf("Basic estimate: PSNR= %5.2f RMSE= %6.3f\n", psnr_basic, rmse_basic);
            printf("Final estimate: PSNR= %5.2f RMSE= %6.3f\n", psnr, rmse);
        }

        FILE* file = fopen(meas_path.c_str(), "w");
        if(file)
        {
            fprintf(file,
                    "-sigma = %f\n-PSNR_basic = %f\n-RSME_basic = %f\n"
                    "-PSNR = %f\n-RMSE = %f\n",
                    fSigma, psnr_basic, rmse_basic, psnr, rmse);
            fclose(file);
        } else
            fprintf(stderr, "Can't open %s\n", meas_path.c_str());

        if(noisy_path != "")
            vid_noisy->saveVideo(noisy_path, firstFrame, frameStep);
    }
    //! save noisy, denoised and differences videos
    if(verbose)
        printf("Save output videos...\n");
    if(basic_path != "")
        vid_basic->saveVideo(basic_path, firstFrame, frameStep);
    if(prms_2.k)
        vid_denoised->saveVideo(final_path, firstFrame, frameStep);
    //! Compute Difference
    if(diffi_path != "")
    {
        vid_diff = vid_basic; // Save allocation of big buffer
        VideoUtils::computeDiff(*vid_noisy, *vid_denoised, *vid_diff, fSigma);
        vid_diff->saveVideo(diffi_path, firstFrame, frameStep);
    }

    return EXIT_SUCCESS;
}
