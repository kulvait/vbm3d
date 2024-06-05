/*
 * Copyright (c) 2015, Pablo Arias <pariasm@gmail.com>
 * All rights reserved.
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later
 * version. You should have received a copy of this license along
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef LIB_VIDEOT_HPP_INCLUDED
#define LIB_VIDEOT_HPP_INCLUDED

#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

#include "Utilities.h"
#include "mt19937ar.h"
#include <signal.h>

/**
 * @brief Structure containing size informations of a video.
 *
 * @param width    : width of the image;
 * @param height   : height of the image;
 * @param channels : number of channels in the image;
 * @param frames   : number of frames in the video;
 * @param wh       : equal to width * height. Provided for convenience;
 * @param whc      : equal to width * height * channels. Provided for convenience.
 * @param whcf     : equal to width * height * frames * channels. Provided for convenience.
 * @param whf      : equal to width * height * frames. Provided for convenience.
 **/
struct VideoSize
{
    uint64_t width;
    uint64_t height;
    uint64_t frames;
    uint64_t channels;
    uint64_t wh;
    uint64_t whc;
    uint64_t whcf;
    uint64_t whf;

    //! Constuctors
    VideoSize(void)
        : width(0)
        , height(0)
        , frames(0)
        , channels(0)
    {
        update_fields();
    }

    VideoSize(uint64_t w, uint64_t h, uint64_t f, uint64_t c)
        : width(w)
        , height(h)
        , frames(f)
        , channels(c)
    {
        update_fields();
    }

    //! Comparison operators
    inline bool operator==(const VideoSize& sz) const
    {
        return (width == sz.width && height == sz.height && channels == sz.channels
                && frames == sz.frames);
    }

    inline bool operator!=(const VideoSize& sz) const { return !operator==(sz); }

    //! Updates products of dimensions
    inline void update_fields(void)
    {
        wh = width * height;
        whc = wh * channels;
        whcf = whc * frames;
        whf = wh * frames;
    }

    //! Returns index
    inline uint64_t index(uint64_t x, uint64_t y, uint64_t t, uint64_t c) const
    {
        // assert(x < width && y < height && t < frames && c < channels);
        return t * whc + c * wh + y * width + x;
    }

    //! Returns index assuming the video has one channel
    inline uint64_t index(uint64_t x, uint64_t y, uint64_t t) const
    {
        assert(x < width && y < height && t < frames);
        return t * wh + y * width + x;
    }

    //! Compute coordinates from index
    inline void coords(uint64_t idx, uint64_t& x, uint64_t& y, uint64_t& t, uint64_t& c) const
    {
        assert(idx < whcf);
        t = (idx) / whc;
        c = (idx % whc) / wh;
        y = (idx % wh) / width;
        x = (idx % width);
    }

    //! Compute coordinates from index assuming the video has one channel
    inline void coords(uint64_t idx, uint64_t& x, uint64_t& y, uint64_t& t) const
    {
        assert(idx < whf);
        t = (idx) / wh;
        y = (idx % wh) / width;
        x = (idx % width);
    }
};

/**
 * @brief A video class template with very basic functionalities.
 *
 * NOTE: should not be used with T = bool, since current implementation
 * relies on std::vector, and std::vector<bool> cannot return a non-
 * constant reference to an element of the array.
 *
 * @param sz       : VideoSize structure with size of the video;
 * @param data     : pointer to an std::vector<T> containing the data
 **/
template <class T>
class Video
{
public:
    //! Size
    VideoSize sz;

    //! Data
    std::vector<T> data;

    //! Constructors
    Video(void); //< empty
    Video(const Video& i_in); //< copy
    Video(const std::string i_pathToFiles,
          uint64_t i_firstFrame,
          uint64_t i_lastFrame,
          uint64_t i_frameStep = 1); //< from filename
    Video(uint64_t i_width,
          uint64_t i_height,
          uint64_t i_frames,
          uint64_t i_channels = 1); //< alloc
    Video(uint64_t i_width,
          uint64_t i_height,
          uint64_t i_frames,
          uint64_t i_channels,
          T val); //< init
    Video(const VideoSize& i_size); //< alloc
    Video(const VideoSize& i_size, T val); //< init

    //! Destructor
    ~Video(void){};

    void clear(void);
    void zero(void) { std::fill(data.begin(), data.end(), T(0)); }
    void resize(uint64_t i_width, uint64_t i_height, uint64_t frames, uint64_t i_channels = 1);
    void resize(const VideoSize& i_size);

    void add(const Video<T>& i_vid)
    {
        if(sz != i_vid.sz)
            throw std::runtime_error("Video<T>::add: videos have different sizes");
        std::transform(data.begin(), data.end(), i_vid.data.begin(), data.begin(), std::plus<T>());
    }

    void divide(const Video<T>& vid_denominator, const Video<T>& vid_zerodivision)
    {
        if(sz != vid_denominator.sz || sz != vid_zerodivision.sz)
        {
            throw std::runtime_error("Video<T>::divide: videos have different sizes");
        }
        for(uint64_t k = 0; k < sz.whcf; k++)
        {
            data[k] = vid_denominator(k) == 0 ? vid_zerodivision(k) : data[k] / vid_denominator(k);
        }
    }

    //! Read/write pixel access ~ inline for efficiency
    T& operator()(uint64_t idx); //< from coordinates
    T& operator()(uint64_t x, uint64_t y, uint64_t t, uint64_t c = 0); //< from coordinates

    //! Read only pixel access ~ inline for efficiency
    T operator()(uint64_t idx) const; //< from index
    T operator()(uint64_t x, uint64_t y, uint64_t t, uint64_t c = 0) const; //< from coordinates

    //! Pixel access with special boundary conditions
    T& getPixelSymmetric(int x, int y, int t, uint64_t c = 0);
    T getPixelSymmetric(int x, int y, int t, uint64_t c = 0) const;

    uint64_t getIndexSymmetric(int x, int y, int t, uint64_t c = 0) const;

    //! I/O
    void loadVideo(const std::string i_pathToFiles,
                   uint64_t i_firstFrame,
                   uint64_t i_lastFrame,
                   uint64_t i_frameStep = 1);
    void loadFullFlow(const std::string i_pathToFiles,
                      uint64_t i_firstFrame,
                      uint64_t i_lastFrame,
                      uint64_t i_frameStep = 1);
    void saveVideo(const std::string i_pathToFiles,
                   uint64_t i_firstFrame,
                   uint64_t i_frameStep = 1,
                   T i_pmin = 0,
                   T i_pmax = 255) const;
    void saveVideoAscii(const std::string i_prefix,
                        uint64_t i_firstFrame,
                        uint64_t i_frameStep = 1) const;

    void transformVideoToBayer(const Video<T>& input);
    void transformVideoFromBayer(const Video<T>& input);

    std::vector<T> dct_shuffle_video(int initFrame, int lastFrame);
};

// Implementations

template <class T>
Video<T>::Video(void)
    : sz()
    , data(0)
{
}

template <class T>
Video<T>::Video(const Video& i_in)
    : sz(i_in.sz)
    , data(i_in.data)
{
}

template <class T>
Video<T>::Video(const std::string i_pathToFiles,
                uint64_t i_firstFrame,
                uint64_t i_lastFrame,
                uint64_t i_frameStep)
    : sz()
    , data(0)
{
    loadVideo(i_pathToFiles, i_firstFrame, i_lastFrame, i_frameStep);
}

template <class T>
Video<T>::Video(uint64_t i_width, uint64_t i_height, uint64_t i_frames, uint64_t i_channels)
    : sz(i_width, i_height, i_frames, i_channels)
    , data(sz.whcf)
{
}

template <class T>
Video<T>::Video(uint64_t i_width, uint64_t i_height, uint64_t i_frames, uint64_t i_channels, T val)
    : sz(i_width, i_height, i_frames, i_channels)
    , data(sz.whcf, val)
{
}

template <class T>
Video<T>::Video(const VideoSize& i_size, T val)
    : sz(i_size)
    , data(sz.whcf, val)
{
}

template <class T>
Video<T>::Video(const VideoSize& i_size)
    : sz(i_size)
    , data(sz.whcf)
{
}

template <class T>
void Video<T>::clear(void)
{
    sz.width = 0;
    sz.height = 0;
    sz.frames = 0;
    sz.channels = 0;
    sz.update_fields();
    data.clear();
}

template <class T>
void Video<T>::resize(const VideoSize& i_size)
{
    if(sz != i_size)
    {
        clear();
        sz = i_size;
        data.resize(sz.whcf, T(0));
    }
}

template <class T>
void Video<T>::resize(uint64_t i_width, uint64_t i_height, uint64_t i_frames, uint64_t i_channels)
{
    resize(VideoSize(i_width, i_height, i_frames, i_channels));
}

template <class T>
void Video<T>::loadVideo(const std::string i_pathToFiles,
                         uint64_t i_firstFrame,
                         uint64_t i_lastFrame,
                         uint64_t i_frameStep)
{
    throw std::runtime_error("Video<T>::loadVideo(...) is only implemented "
                             "for T = float");
}

template <>
void Video<float>::loadVideo(const std::string i_pathToFiles,
                             uint64_t i_firstFrame,
                             uint64_t i_lastFrame,
                             uint64_t i_frameStep);

template <class T>
std::vector<T> Video<T>::dct_shuffle_video(int i, int f)
{
    throw std::runtime_error("Video<T>::dct_shuffle_video(...) is only implemented "
                             "for T = float");
}

template <>
std::vector<float> Video<float>::dct_shuffle_video(int i, int f);

template <class T>
void Video<T>::saveVideo(const std::string i_pathToFiles,
                         uint64_t i_firstFrame,
                         uint64_t i_frameStep,
                         T i_pmin,
                         T i_pmax) const
{
    throw std::runtime_error("Video<T>::saveVideo(...) is only implemented "
                             "for T = float");
}

template <>
void Video<float>::saveVideo(const std::string i_pathToFiles,
                             uint64_t i_firstFrame,
                             uint64_t i_frameStep,
                             float i_pmin,
                             float i_pmax) const;

template <class T>
void Video<T>::transformVideoFromBayer(const Video<T>& input)
{
    throw std::runtime_error("Video<T>::transformVideoFromBayer(...) is only implemented "
                             "for T = float");
}
template <>
void Video<float>::transformVideoFromBayer(const Video<float>& input);

template <class T>
void Video<T>::transformVideoToBayer(const Video<T>& input)
{
    throw std::runtime_error("Video<T>::transformVideoToBayer(...) is only implemented "
                             "for T = float");
}
template <>
void Video<float>::transformVideoToBayer(const Video<float>& input);

template <class T>
void Video<T>::saveVideoAscii(const std::string i_prefix,
                              uint64_t i_firstFrame,
                              uint64_t i_frameStep) const
{
    throw std::runtime_error("Video<T>::saveVideoAscii(...) is only implemented "
                             "for T = float");
}

template <>
void Video<float>::saveVideoAscii(const std::string i_prefix,
                                  uint64_t i_firstFrame,
                                  uint64_t i_frameStep) const;

template <class T>
inline T& Video<T>::operator()(uint64_t idx)
{
    assert(idx < sz.whcf);
    return data[idx];
}

template <class T>
inline T& Video<T>::operator()(uint64_t x, uint64_t y, uint64_t t, uint64_t c)
{
    return data[sz.index(x, y, t, c)];
}

template <class T>
inline T Video<T>::operator()(uint64_t idx) const
{
    assert(idx < sz.whcf);
    return data[idx];
}

template <class T>
inline T Video<T>::operator()(uint64_t x, uint64_t y, uint64_t t, uint64_t c) const
{
    return data[sz.index(x, y, t, c)];
}

template <class T>
inline T& Video<T>::getPixelSymmetric(int x, int y, int t, uint64_t c)
{
    // NOTE: assumes that -width+1 < x < 2*(width -1)
    assert(-(int)sz.width < x && x < 2 * (int)sz.width - 1 && -(int)sz.height < y
           && y < 2 * (int)sz.height - 1 && -(int)sz.frames < t && t < 2 * (int)sz.frames - 1);
    // symmetrize
    x = (x < 0) ? -x : (x >= (int)sz.width) ? 2 * (int)sz.width - 2 - x : x;
    y = (y < 0) ? -y : (y >= (int)sz.height) ? 2 * (int)sz.height - 2 - y : y;
    t = (t < 0) ? -t : (t >= (int)sz.frames) ? 2 * (int)sz.frames - 2 - t : t;

    return data[sz.index(x, y, t, c)];
}

template <class T>
inline T Video<T>::getPixelSymmetric(int x, int y, int t, uint64_t c) const
{
    // NOTE: assumes that -width+1 < x < 2*(width -1)
    assert(-(int)sz.width < x && x < 2 * (int)sz.width - 1 && -(int)sz.height < y
           && y < 2 * (int)sz.height - 1 && -(int)sz.frames < t && t < 2 * (int)sz.frames - 1);
    // symmetrize
    x = (x < 0) ? -x : (x >= (int)sz.width) ? 2 * (int)sz.width - 2 - x : x;
    y = (y < 0) ? -y : (y >= (int)sz.height) ? 2 * (int)sz.height - 2 - y : y;
    t = (t < 0) ? -t : (t >= (int)sz.frames) ? 2 * (int)sz.frames - 2 - t : t;

    return data[sz.index(x, y, t, c)];
}

template <class T>
inline uint64_t Video<T>::getIndexSymmetric(int x, int y, int t, uint64_t c) const
{
    // NOTE: assumes that -width+1 < x < 2*(width -1)
    assert(-(int)sz.width < x && x < 2 * (int)sz.width - 1 && -(int)sz.height < y
           && y < 2 * (int)sz.height - 1 && -(int)sz.frames < t && t < 2 * (int)sz.frames - 1);
    // symmetrize
    x = (x < 0) ? -x : (x >= (int)sz.width) ? 2 * (int)sz.width - 2 - x : x;
    y = (y < 0) ? -y : (y >= (int)sz.height) ? 2 * (int)sz.height - 2 - y : y;
    t = (t < 0) ? -t : (t >= (int)sz.frames) ? 2 * (int)sz.frames - 2 - t : t;

    return sz.index(x, y, t, c);
}

//! Utilities for video
namespace VideoUtils {
/**
 * @brief Structure to store the position of a rectangular crop. It also has
 * data to describe the position of the crop when it corresponds to a rectangular
 * tiling (potentially with an added border) of a video. The tiles in the tiling
 * do not overlap, but the crop can correspond to a tile with an added border.
 * origin and ending encode the crop coordinates where as tile_origin and
 * tile_ending correspond to the tile coordinates (tile coordinates are
 * contained in the crop).
 *
 * @param origin_x  : x coordinate of top-left-front corner of crop
 * @param origin_y  : y coordinate of top-left-front corner of crop
 * @param origin_t  : t coordinate of top-left-front corner of crop
 *
 * @param ending_x  : x coordinate of bottom-right-back corner of crop
 * @param ending_y  : y coordinate of bottom-right-back corner of crop
 * @param ending_t  : t coordinate of bottom-right-back corner of crop
 *
 * @param source_sz : size of source video
 *
 * @params tile_x   : x index of tile (0 <= tile_x < ntiles_x)
 * @params tile_y   : y index of tile (0 <= tile_y < ntiles_y)
 * @params tile_t   : t index of tile (0 <= tile_t < ntiles_t)

 * @params ntiles_x : total number of tiles in x direction
 * @params ntiles_y : total number of tiles in y direction
 * @params ntiles_t : total number of tiles in t direction

 * @params tile_origin_x : x coordinate of top-left-front corner of tile
 * @params tile_origin_y : y coordinate of top-left-front corner of tile
 * @params tile_origin_t : t coordinate of top-left-front corner of tile

 * @params tile_ending_x : x coordinate of bottom-right-back corner of tile
 * @params tile_ending_y : y coordinate of bottom-right-back corner of tile
 * @params tile_ending_t : t coordinate of bottom-right-back corner of tile
 *
 *
 **/
struct CropPosition
{
    int origin_x;
    int origin_y;
    int origin_t;

    int ending_x;
    int ending_y;
    int ending_t;

    VideoSize source_sz;

    int tile_x;
    int tile_y;
    int tile_t;

    int ntiles_x;
    int ntiles_y;
    int ntiles_t;

    int tile_origin_x;
    int tile_origin_y;
    int tile_origin_t;

    int tile_ending_x;
    int tile_ending_y;
    int tile_ending_t;
};

/**
 * @brief add noise to video.
 *
 * @param i_vid : original noise-free image;
 * @param o_vidNoisy = vid + noise;
 * @param p_sigma : standard deviation of the noise.
 *
 * @return none.
 **/
template <class T>
void addNoise(Video<T> const& i_vid,
              Video<T>& o_vidNoisy,
              const float p_sigma,
              const bool p_verbose = false)
{
    if(p_verbose)
        printf("Add noise with sigma = %g\n", p_sigma);

    //! Initialization
    o_vidNoisy = i_vid;
    // mt_init_genrand((uint64_t long int) time (NULL) +
    //                (uint64_t long int) getpid());
    mt_init_genrand((uint64_t)time(NULL));
    // mt_init_genrand(0); printf("\x1b[33;1mWarning:\x1b[0m random generator seed is 0\n");

    //! Add noise
    for(uint64_t k = 0; k < i_vid.sz.whcf; k++)
    {
        const double a = mt_genrand_res53();
        const double b = mt_genrand_res53();
        o_vidNoisy(k) += (T)(p_sigma * sqrtl(-2.0l * log(a)) * cos(2.0l * M_PI * b));
    }
}

/**
 * @brief Compute PSNR and RMSE between i_vid1 and i_vid2
 *
 * @param i_vid1 : video 1;
 * @param i_vid2 : video 2;
 * @param o_psnr  : will contain the PSNR;
 * @param o_rmse  : will contain the RMSE;
 *
 * @return none.
 **/
template <class T>
void computePSNR(Video<T> const& i_vid1, Video<T> const& i_vid2, double& o_psnr, double& o_rmse)
{
    if(i_vid1.sz != i_vid2.sz)
        throw std::runtime_error("VideoUtils::computePSNR: videos have different sizes");

    double sum = 0.f;
    for(uint64_t k = 0; k < i_vid1.sz.whcf; k++)
    {
        sum += ((float)clip(i_vid1(k), 0, 255) - (float)clip(i_vid2(k), 0, 255))
            * ((float)clip(i_vid1(k), 0, 255) - (float)clip(i_vid2(k), 0, 255));
    }

    o_rmse = sqrtf(sum / (float)i_vid1.sz.whcf);
    o_psnr = 20.f * log10f(255.f / o_rmse);

    return;
}

/**
 * @brief Compute a difference video between i_vid1 and i_vid2.
 *
 * @param i_vid1: reference video;
 * @param i_vid2: video to compare;
 * @param o_vidDiff: will contain the difference;
 * @param p_sigma: standard deviation of the noise;
 * @param p_min, p_max : range of data (usually [0, 255]);
 *
 * @return none.
 **/
template <class T>
void computeDiff(Video<T> const& i_vid1,
                 Video<T> const& i_vid2,
                 Video<float>& o_vidDiff,
                 const float p_sigma,
                 const T p_min = 0.f,
                 const T p_max = 255.f)
{
    if(i_vid1.sz != i_vid2.sz)
        throw std::runtime_error("VideoUtils::computeDiff: videos have different sizes");

    o_vidDiff.resize(i_vid1.sz);
    for(uint64_t k = 0; k < i_vid1.sz.whcf; k++)
    {
        float value = ((float)i_vid1(k) - (float)i_vid2(k)) + (p_max - p_min) / 2.f;
        o_vidDiff(k) = clip(value, p_min, p_max);
    }

    return;
}

/**
 * @brief 'Generalized' croping of a video (cropped video may be larger than original).
 *
 * @param i_vid1 : original video;
 * @param o_vid2 : output video, already allocated to desired size;
 * @param i_origin2 : vid1 coordinates of vid2 origin. Origin coordinates
 * larger than corresponding vid1 dimension are redefined to center the crop
 * in that dimension.
 *
 * @return none.
 **/
template <class T>
void crop(Video<T> const& i_vid1,
          Video<T>& o_vid2,
          int p_origin_t = INT_MAX,
          int p_origin_x = INT_MAX,
          int p_origin_y = INT_MAX)
{
    assert(o_vid2.sz.channels == i_vid1.sz.channels);

    //! Redefine invalid origin coordinates to default (centered crop)
    if(p_origin_t > (int)i_vid1.sz.frames)
        p_origin_t = ((int)i_vid1.sz.frames - (int)o_vid2.sz.frames) / 2;
    if(p_origin_x > (int)i_vid1.sz.width)
        p_origin_x = ((int)i_vid1.sz.width - (int)o_vid2.sz.width) / 2;
    if(p_origin_y > (int)i_vid1.sz.height)
        p_origin_y = ((int)i_vid1.sz.height - (int)o_vid2.sz.height) / 2;

    // TODO: more efficient implementation
    for(int f = 0; f < o_vid2.sz.frames; f++)
        for(uint64_t c = 0; c < o_vid2.sz.channels; c++)
            for(int y = 0; y < o_vid2.sz.height; y++)
                for(int x = 0; x < o_vid2.sz.width; x++)
                    o_vid2(x, y, f, c) = i_vid1.getPixelSymmetric(x + p_origin_x, y + p_origin_y,
                                                                  f + p_origin_t, c);
}

/**
 * @brief 'Generalized' croping of a video (cropped video may be larger than original).
 *
 * @param i_vid1 : original video;
 * @param o_vid2 : output video, already allocated to desired size;
 * @param i_origin2 : vid1 coordinates of vid2 origin. Origin coordinates
 * larger than corresponding vid1 dimension are redefined to center the crop
 * in that dimension.
 *
 * @return none.
 **/
template <class T>
void crop(Video<T> const& i_vid1, Video<T>& o_vid2, const int* const p_origin)
{
    crop(i_vid1, o_vid2, p_origin[2], p_origin[0], p_origin[1]);
}

/**
 * @brief 'Generalized' croping of a video (cropped video may be larger than original).
 *
 * @param i_vid1 : original video;
 * @param o_vid2 : output video, already allocated to desired size;
 * @param i_origin2 : vid1 coordinates of vid2 origin. Origin coordinates
 * larger than corresponding vid1 dimension are redefined to center the crop
 * in that dimension.
 *
 * @return none.
 **/
template <class T>
void crop(Video<T> const& i_vid1, Video<T>& o_vid2, CropPosition const& p_crop)
{
    //! Resize output video
    VideoSize cropSize;
    cropSize.width = p_crop.ending_x - p_crop.origin_x;
    cropSize.height = p_crop.ending_y - p_crop.origin_y;
    cropSize.frames = p_crop.ending_t - p_crop.origin_t;
    cropSize.channels = i_vid1.sz.channels;
    cropSize.update_fields();

    o_vid2.resize(cropSize);

    crop(i_vid1, o_vid2, p_crop.origin_t, p_crop.origin_x, p_crop.origin_y);
}

/**
 * @brief Add boundaries by symmetry
 *
 * @param io_vid : original image;
 * @param io_vidSym : contain io_im symmetrized;
 *
 * @return none.
 **/
template <class T>
void addBorder(Video<T> const& i_vid1,
               Video<T>& o_vid2,
               const uint64_t p_borderSize,
               const bool p_isForward)
{
    //! Sizes
    const uint64_t w2 = i_vid1.sz.width + (p_isForward ? 2 * p_borderSize : -2 * p_borderSize);
    const uint64_t h2 = i_vid1.sz.height + (p_isForward ? 2 * p_borderSize : -2 * p_borderSize);
    // const uint64_t f2 = i_vid1.sz.frames + (p_isForward ? 2*p_borderSize : - 2*p_borderSize);
    //! For VBM3D we only use 2D patches, we don't need any border for the temporal dimension
    const uint64_t f2 = i_vid1.sz.frames;
    const uint64_t ch = i_vid1.sz.channels;

    //! Position of vid2 origin in vid1 coordinates
    const int tx = p_isForward ? -p_borderSize : p_borderSize;
    const int ty = p_isForward ? -p_borderSize : p_borderSize;
    // const int tf = p_isForward ? -p_borderSize : p_borderSize;
    //! For VBM3D we only use 2D patches, we don't need any border for the temporal dimension
    const int tf = 0;

    //! Resize output image, if necessary
    o_vid2.resize(w2, h2, f2, ch);

    //! Call generalized crop
    crop(i_vid1, o_vid2, tf, tx, ty);
}

/**
 * @brief Transform the color space of an video, from RGB to YUV, or vice-versa.
 *
 * @param io_vid: image on which the transform will be applied;
 * @param p_isForward: if true, go from RGB to YUV, otherwise go from YUV to RGB.
 *
 * @return none.
 **/
template <class T>
void transformColorSpace(Video<T>& io_vid, const bool p_isForward)
{
    //! If the image as only one channel, do nothing
    if(io_vid.sz.channels == 1)
        return;

    //! Initialization
    // const uint64_t width  = io_vid.sz.width;
    // const uint64_t height = io_vid.sz.height;
    const uint64_t chnls = io_vid.sz.channels;
    const uint64_t wh = io_vid.sz.wh;

    for(int f = 0; f < io_vid.sz.frames; f++)
        if(p_isForward) //< RGB to YUV
        {
            if(chnls == 3)
            {
                const uint64_t red = f * io_vid.sz.whc;
                const uint64_t green = red + wh;
                const uint64_t blue = green + wh;
                // Following constants were precomputed
                // const float a = 1.f / sqrtf(3.f);
                // const float b = 1.f / sqrtf(2.f);
                // const float c = 2.f * a * sqrtf(2.f);

                float yuv[3];
                for(uint64_t k = 0; k < wh; k++)
                {
                    //! Y
                    yuv[0] = 0.299f * io_vid(k + red) + 0.587f * io_vid(k + green)
                        + 0.114f * io_vid(k + blue);
                    //! U
                    yuv[1] = -0.14713f * io_vid(k + red) - 0.28886f * io_vid(k + green)
                        + 0.436f * io_vid(k + blue);
                    //! V
                    yuv[2] = 0.615f * io_vid(k + red) - 0.51498f * io_vid(k + green)
                        - 0.10001f * io_vid(k + blue);

                    // yuv[0] = a * ((float)io_vid(k + red) + (float)io_vid(k + green) +
                    // (float)io_vid(k + blue)); yuv[1] = b * ((float)io_vid(k + red) -
                    // (float)io_vid(k + blue)); yuv[2] = c * (0.25f * (float)io_vid(k + red ) -
                    // 0.5f
                    // * (float)io_vid(k + green)
                    //		      + 0.25f * (float)io_vid(k + blue));

                    io_vid(k + red) = (T)yuv[0];
                    io_vid(k + green) = (T)yuv[1];
                    io_vid(k + blue) = (T)yuv[2];
                }
            } else //< chnls == 4
            {
                const uint64_t Gr = f * io_vid.sz.whc;
                const uint64_t R = Gr + wh;
                const uint64_t B = R + wh;
                const uint64_t Gb = B + wh;
                const float a = 0.5f;
                const float b = 1.f / sqrtf(2.f);

                float wtf[4];
                for(uint64_t k = 0; k < wh; k++)
                {
                    wtf[0] = a
                        * ((float)io_vid(k + Gr) + (float)io_vid(k + R) + (float)io_vid(k + B)
                           + (float)io_vid(k + Gb));
                    wtf[1] = b * ((float)io_vid(k + R) - (float)io_vid(k + B));
                    wtf[2] = a
                        * (-(float)io_vid(k + Gr) + (float)io_vid(k + R) + (float)io_vid(k + B)
                           - (float)io_vid(k + Gb));
                    wtf[3] = b * (-(float)io_vid(k + Gr) + (float)io_vid(k + Gb));
                    io_vid(k + Gr) = (T)wtf[0];
                    io_vid(k + R) = (T)wtf[1];
                    io_vid(k + B) = (T)wtf[2];
                    io_vid(k + Gb) = (T)wtf[3];
                }
            }
        } else //< YUV to RGB
        {
            if(chnls == 3)
            {
                const uint64_t red = f * io_vid.sz.whc;
                const uint64_t green = red + wh;
                const uint64_t blue = green + wh;
                // const float a = 1.f / sqrtf(3.f);
                // const float b = 1.f / sqrtf(2.f);
                // const float c = a / b;

                float rgb[3];
                for(uint64_t k = 0; k < wh; k++)
                {
                    //! Red   channel
                    rgb[0] = io_vid(k + red) + 1.13983f * io_vid(k + blue);
                    //! Green channel
                    rgb[1] = io_vid(k + red) - 0.39465f * io_vid(k + green)
                        - 0.5806f * io_vid(k + blue);
                    //! Blue  channel
                    rgb[2] = io_vid(k + red) + 2.03211f * io_vid(k + green);

                    // rgb[0] = a * (float)io_vid(k + red) + b * (float)io_vid(k + green)
                    //                             + c * 0.5f * (float)io_vid(k + blue );
                    // rgb[1] = a * (float)io_vid(k + red) - c * (float)io_vid(k + blue );
                    // rgb[2] = a * (float)io_vid(k + red) - b * (float)io_vid(k + green)
                    //                      + c * 0.5f * (float)io_vid(k + blue );
                    io_vid(k + red) = (T)rgb[0];
                    io_vid(k + green) = (T)rgb[1];
                    io_vid(k + blue) = (T)rgb[2];
                }
            } else //! chnls == 4
            {
                const uint64_t Gr = f * io_vid.sz.whc;
                const uint64_t R = Gr + wh;
                const uint64_t B = R + wh;
                const uint64_t Gb = B + wh;
                const float a = 0.5f;
                const float b = 1.f / sqrtf(2.f);

                float wtf[4];
                for(uint64_t k = 0; k < wh; k++)
                {
                    wtf[0] = a * (float)io_vid(k + Gr) - a * (float)io_vid(k + B)
                        - b * (float)io_vid(k + Gb);
                    wtf[1] = a * (float)io_vid(k + Gr) + b * (float)io_vid(k + R)
                        + a * (float)io_vid(k + B);
                    wtf[2] = a * (float)io_vid(k + Gr) - b * (float)io_vid(k + R)
                        + a * (float)io_vid(k + B);
                    wtf[3] = a * (float)io_vid(k + Gr) - a * (float)io_vid(k + B)
                        + b * (float)io_vid(k + Gb);
                    io_vid(k + Gr) = (T)wtf[0];
                    io_vid(k + R) = (T)wtf[1];
                    io_vid(k + B) = (T)wtf[2];
                    io_vid(k + Gb) = (T)wtf[3];
                }
            }
        }
}

/**
 * @brief Subdivide a video into small sub-videos. This version
 * does not add a border to the video, resulting in parts which
 * might have different sizes.
 *
 * @param i_video : image to subdivide;
 * @param o_videoSub : will contain all sub-videos;
 * @param o_crops : will store position of the crops;
 * @param p_N : boundary around sub-videos;
 * @param p_nb : number of sub-videos wanted. Need to be a power of 2.
 *
 * @return none.
 **/
template <class T>
void subDivideTight(Video<T> const& i_vid,
                    std::vector<CropPosition>& o_crops,
                    const int p_N,
                    const int p_nb)
{
    /* FIXME current version splits the video only spatially.
     *       The reason is to mantain consistency with Marc's
     *       code. For its proper extension to video, we need
     *       to determine how to split the video in space and
     *       time. */

    //! Determine number of sub-images
    uint64_t u_nW, u_nH; // FIXME problem with uint64_t and int
    determineFactor((uint64_t)p_nb, u_nW, u_nH);
    int nW = (int)u_nW;
    int nH = (int)u_nH;

    const int wTmp = ceil(float(i_vid.sz.width) / float(nW)); // sizes w/out
    const int hTmp = ceil(float(i_vid.sz.height) / float(nH)); //     borders

    o_crops.resize(p_nb);
    for(int p = 0, n = 0; p < nH; p++)
        for(int q = 0; q < nW; q++, n++)
        {
            //! Set crop information
            o_crops[n].source_sz = i_vid.sz;

            //! The origin is shifted -p_N to account for the subimage border
            o_crops[n].origin_x = std::max(0, q * wTmp - p_N);
            o_crops[n].origin_y = std::max(0, p * hTmp - p_N);
            o_crops[n].origin_t = 0;

            //! The origin is shifted p_N to account for the subimage border
            o_crops[n].ending_x = std::min((int)i_vid.sz.width, (q + 1) * wTmp + p_N);
            o_crops[n].ending_y = std::min((int)i_vid.sz.height, (p + 1) * hTmp + p_N);
            o_crops[n].ending_t = i_vid.sz.frames;

            //! Add information about the tiling
            o_crops[n].tile_x = q;
            o_crops[n].tile_y = p;
            o_crops[n].tile_t = 0;

            o_crops[n].ntiles_x = nW;
            o_crops[n].ntiles_y = nH;
            o_crops[n].ntiles_t = 1;

            o_crops[n].tile_origin_x = q * wTmp;
            o_crops[n].tile_origin_y = p * wTmp;
            o_crops[n].tile_origin_t = 0;

            o_crops[n].tile_ending_x = std::min((int)i_vid.sz.width, (q + 1) * wTmp);
            o_crops[n].tile_ending_y = std::min((int)i_vid.sz.height, (p + 1) * hTmp);
            o_crops[n].tile_ending_t = i_vid.sz.frames;
        }

    return;
}

/**
 * @brief Subdivide a video into small sub-videos
 *
 * @param i_video : image to subdivide;
 * @param o_videoSub : will contain all sub-videos;
 * @param o_crops : will store position of the crops;
 * @param p_N : boundary around sub-videos;
 * @param p_nb : number of sub-videos wanted. Need to be a power of 2.
 *
 * @return none.
 **/
template <class T>
void subDivide(Video<T> const& i_vid,
               std::vector<Video<T>>& o_vidSub,
               std::vector<CropPosition>& o_crops,
               const uint64_t p_N,
               const uint64_t p_nb)
{
    /* FIXME current version splits the video only spatially.
     *       The reason is to mantain consistency with Marc's
     *       code. For its proper extension to video, we need
     *       to determine how to split the video in space and
     *       time. */

    //! Determine number of sub-images
    uint64_t u_nW, u_nH; // FIXME problem with uint64_t and int
    determineFactor((uint64_t)p_nb, u_nW, u_nH);
    int nW = (int)u_nW;
    int nH = (int)u_nH;

    const int wTmp = ceil(float(i_vid.sz.width) / float(nW)); // sizes w/out
    const int hTmp = ceil(float(i_vid.sz.height) / float(nH)); //     borders

    //! Obtain sub-images
    VideoSize imSubSize;
    imSubSize.width = wTmp + 2 * p_N; // each sub-image has border
    imSubSize.height = hTmp + 2 * p_N;
    imSubSize.frames = i_vid.sz.frames; // NOTE: same frames as original
    imSubSize.channels = i_vid.sz.channels;
    imSubSize.update_fields();

    o_crops.resize(p_nb);
    o_vidSub.resize(p_nb);
    for(int p = 0, n = 0; p < nH; p++)
        for(int q = 0; q < nW; q++, n++)
        {
            o_vidSub[n].resize(imSubSize);

            //! The origin is shifted -p_N to account for the subimage border
            int origin[3] = { q * wTmp - p_N, p * hTmp - p_N, 0 };

            //! Crop using symmetric boundary conditions
            VideoUtils::crop(i_vid, o_vidSub[n], origin);

            //! Set crop information
            o_crops[n].source_sz = i_vid.sz;
            o_crops[n].origin_x = origin[0];
            o_crops[n].origin_y = origin[1];
            o_crops[n].origin_t = origin[2];

            o_crops[n].ending_x = origin[0] + o_vidSub[n].sz.width;
            o_crops[n].ending_y = origin[1] + o_vidSub[n].sz.height;
            o_crops[n].ending_t = origin[2] + o_vidSub[n].sz.frames;

            //! Add information about the tiling
            o_crops[n].tile_x = q;
            o_crops[n].tile_y = p;
            o_crops[n].tile_t = 0;

            o_crops[n].ntiles_x = nW;
            o_crops[n].ntiles_y = nH;
            o_crops[n].ntiles_t = 1;

            o_crops[n].tile_origin_x = q * wTmp;
            o_crops[n].tile_origin_y = p * wTmp;
            o_crops[n].tile_origin_t = 0;

            o_crops[n].tile_ending_x = std::min((int)i_vid.sz.width, (q + 1) * wTmp);
            o_crops[n].tile_ending_y = std::min((int)i_vid.sz.height, (p + 1) * hTmp);
            o_crops[n].tile_ending_t = i_vid.sz.frames;
        }

    return;
}

/**
 * @brief Subdivide a video into small sub-videos
 *
 * @param i_video : image to subdivide;
 * @param o_videoSub : will contain all sub-videos;
 * @param p_N : boundary around sub-videos;
 * @param p_nb : number of sub-videos wanted. Need to be a power of 2.
 *
 * @return none.
 **/
template <class T>
void subDivide(Video<T> const& i_vid,
               std::vector<Video<T>>& o_vidSub,
               const uint64_t p_N,
               const uint64_t p_nb)
{
    std::vector<CropPosition> tmpCrops;
    subDivide(i_vid, o_vidSub, tmpCrops, p_N, p_nb);
    return;
}

/**
 * @brief Reconstruct an video from its small sub-videos
 *
 * @param o_vid : image to reconstruct;
 * @param i_vidSub : will contain all sub-images;
 * @param p_N : boundary around sub-videos.
 *
 * @return none.
 **/
template <class T>
void subBuild(std::vector<Video<T>> const& i_vidSub, Video<T>& o_vid, const uint64_t p_N)
{
    /* FIXME current version builds a video that has been split
     *       only spatially by subDivide.
     *       The reason is to mantain consistency with Marc's
     *       code. For its proper extension to video, we need
     *       to determine how to split the video in space and
     *       time. */

    assert(i_vidSub.size());
    assert(i_vidSub[0].sz.whcf);
    assert(o_vid.sz.whcf);
    assert(o_vid.sz.frames == i_vidSub[0].sz.frames);
    assert(o_vid.sz.channels == i_vidSub[0].sz.channels);

    //! Determine width and height composition
    uint64_t nW, nH;
    determineFactor(i_vidSub.size(), nW, nH);
    const uint64_t hTmp = i_vidSub[0].sz.height - 2 * p_N;
    const uint64_t wTmp = i_vidSub[0].sz.width - 2 * p_N;

    //! Obtain inner image (containing boundaries)
    // TODO pending decision for video
    for(uint64_t py = 0, n = 0; py < nH * hTmp; py += hTmp)
        for(uint64_t px = 0; px < nW * wTmp; px += wTmp, n++)
        {
            /* Diagram for a 1D image with W = 8, covered
             * with 2 sub images of w = 5, with border 2.
             * Symmetrized pixels are indicated with an s.
             *
             * ori         0  1  2  3  4  5  6  7
             * sub1 s0 s1  2  3  4  5  6 s7 s8
             * sub2                s0 s1  2  3  4 s5 s6 s7 s8
             *
             * px         0*w            1*w       <-- don't exceed 1*w + W-1*w
             *
             * Notation: [px,py] coords on big image of sub-image top-left point
             *           [qx,qy] point on big image
             *           [sx,sy] corresponding point on sub-image
             */
            uint64_t wmax = std::min(wTmp, o_vid.sz.width - px) + p_N;
            uint64_t hmax = std::min(hTmp, o_vid.sz.height - py) + p_N;

            for(uint64_t f = 0; f < o_vid.sz.frames; f++)
                for(uint64_t c = 0; c < o_vid.sz.channels; c++)
                    for(uint64_t sy = p_N, qy = py; sy < hmax; sy++, qy++)
                        for(uint64_t sx = p_N, qx = px; sx < wmax; sx++, qx++)
                            o_vid(qx, qy, f, c) = i_vidSub[n](sx, sy, f, c);
        }

    return;
}

/**
 * @brief Reconstruct an video from its small sub-videos
 *
 * @param o_vid : image to reconstruct;
 * @param i_vidSub : will contain all sub-images;
 * @param p_N : boundary around sub-videos.
 *
 * @return none.
 **/
template <class T>
void subBuildTight(std::vector<Video<T>> const& i_vidSub, Video<T>& o_vid, const int p_N)
{
    /* FIXME current version builds a video that has been split
     *       only spatially by subDivide.
     *       The reason is to mantain consistency with Marc's
     *       code. For its proper extension to video, we need
     *       to determine how to split the video in space and
     *       time. */

    assert(i_vidSub.size());
    assert(i_vidSub[0].sz.whcf);
    assert(o_vid.sz.whcf);
    assert(o_vid.sz.frames == i_vidSub[0].sz.frames);
    assert(o_vid.sz.channels == i_vidSub[0].sz.channels);

    //! Determine width and height composition
    uint64_t nW, nH;
    determineFactor(i_vidSub.size(), nW, nH);
    const int wTmp = ceil(float(o_vid.sz.width) / float(nW)); // sizes w/out
    const int hTmp = ceil(float(o_vid.sz.height) / float(nH)); //     borders

    //! Obtain inner image (containing boundaries)
    // TODO pending decision for video
    for(int p = 0, n = 0; p < nH; p++)
        for(int q = 0; q < nW; q++, n++)
        {
            //! top-left-front corner of crop
            int crop_ori_x = std::max(0, q * wTmp - p_N);
            int crop_ori_y = std::max(0, p * hTmp - p_N);
            // int crop_ori_t = 0;

            //! start of inner crop, removing the boundary
            int ori_x = q * wTmp;
            int ori_y = p * hTmp;
            // int ori_t = 0       ;

            //! end of inner crop
            int end_x = std::min((q + 1) * wTmp, (int)o_vid.sz.width);
            int end_y = std::min((p + 1) * hTmp, (int)o_vid.sz.height);
            // int end_t = 0       ;

            //! start of inner crop, with inner coordinates
            int in_ori_x = ori_x - crop_ori_x;
            int in_ori_y = ori_y - crop_ori_y;
            // int in_ori_t = ori_t - crop_ori_t;

            //			int  in_end_x = (q+1) * wTmp - out_ori_x;
            //			int  in_end_y = (p+1) * hTmp - out_ori_y;
            //			int  in_end_t = i_vid.sz.frames;

            for(uint64_t f = 0; f < o_vid.sz.frames; f++)
                for(uint64_t c = 0; c < o_vid.sz.channels; c++)
                    for(uint64_t sy = in_ori_y, qy = ori_y; qy < end_y; sy++, qy++)
                        for(uint64_t sx = in_ori_x, qx = ori_x; qx < end_x; sx++, qx++)
                            o_vid(qx, qy, f, c) = i_vidSub[n](sx, sy, f, c);
        }

    return;
}
} // namespace VideoUtils

#endif // LIB_VIDEOT_HPP_INCLUDED
