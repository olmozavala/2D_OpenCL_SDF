/* 
 * File:   SignedDistFunc.cpp
 * Author: olmozavala
 * 
 * Created on October 10, 2011, 10:08 AM
 */

#define MAXF 1
#define MAXDPHIDT 2
#define EPS .00001

#include "SignedDistFunc.h"
#include <sstream>
#include "debug.h"
#include "MatrixUtils/MatrixUtils.h"
#include <iomanip> //For setting the precision of the printing floats

#define cutImageW 12
#define cutImageH 16
//#define PRINT 1

template <class T>
inline std::string to_string(char* prev, const T& t, char* post) {
    std::stringstream ss;
    ss << prev << t << post;
    return ss.str();
}

inline std::string appendStr(char* prev, char* post) {
    std::stringstream ss;
    ss << prev << post;
    return ss.str();
}

SignedDistFunc::SignedDistFunc() {
}

SignedDistFunc::SignedDistFunc(const SignedDistFunc& orig) {
}

SignedDistFunc::~SignedDistFunc() {
}


/**
 * Main SDF function using buffers.
 * @param {CLManager*} Pointer the the CL manager object we are using.
 * @param {int} SDFmethod (0 for radial OZ and 1 for Voronoi maurer)
 * @param {cl:buffer} bufmask Is the 
 */
cl::Event SignedDistFunc::runSDFBuf(CLManager* clOrig, int SDFmethod, cl::Buffer& buf_mask,
        cl::Buffer& buf_sdf, int max_warp_size, int width, int height, cl::Event evWrtImg, char* save_path) {

#ifdef DEBUG
    Timer tm_prev(sdf_ts, "Prev");
    tm_prev.start(Timer::SELF);
#endif

    this->save_path = save_path;
    this->buf_mask = buf_mask;
    this->buf_sdf = buf_sdf;
    cl = clOrig;

    buf_size = width*height;

    try {

        context = cl->getContext();
        queue = cl->getQueue();
        program = cl->getProgram();


        this->max_warp_size = max_warp_size;

        this->width = width;
        this->height = height;

        origin = CLManager::getSizeT(0, 0, 0);
        region = CLManager::getSizeT(width, height, 1);

        int tot_grps_x = 0;
        int tot_grps_y = 0;

        CLManager::getGroupSize(max_warp_size, width, height, grp_size_x, grp_size_y, tot_grps_x, tot_grps_y, false);

        dout << "Work group size: " << grp_size_x << " x " << grp_size_y << endl;
        dout << "Number of groups: " << tot_grps_x << " x " << tot_grps_y << endl;

        //This array will contain the average gray value inside and outside the object
        buf_sdf_half = cl::Buffer(*context, CL_MEM_READ_WRITE, buf_size * sizeof (float), NULL, &err);
        buf_sdf_half2 = cl::Buffer(*context, CL_MEM_READ_WRITE, buf_size * sizeof (float), NULL, &err);

#ifdef DEBUG
        queue->finish();
        tm_prev.end();
#endif
        vecWriteImage.push_back(evWrtImg);
        //After running this kernel we should already have the right phi
        switch (SDFmethod) {
            case SDFOZ:
                SDFoz();
                break;
            case SDFVORO:
                //for(int i=0; i<10; i++){
                    evEndSDF = SDFVoroBuf();
                //}
                break;
        }

#ifdef DEBUG
        sdf_ts.dumpTimings();
#endif
    } catch (cl::Error ex) {
        cl->printError(ex);
    }

    return evEndSDF; // Delete is just for testing
}

cl::Event SignedDistFunc::runSDF(CLManager* clOrig, int SDFmethod, cl::Image2D& img_mask, cl::Image2D& img_sdf,
        int max_warp_size, int width, int height, cl::Event evWrtImg, char* save_path) {

    this->save_path = save_path;
    this->img_mask = img_mask;
    this->img_sdf = img_sdf;
    cl = clOrig;

    try {
        context = cl->getContext();
        queue = cl->getQueue();
        program = cl->getProgram();

        array_img_out = new float[width * height * 4];

        this->max_warp_size = max_warp_size;

        this->width = width;
        this->height = height;

        origin = CLManager::getSizeT(0, 0, 0);
        region = CLManager::getSizeT(width, height, 1);

        int tot_grps_x = 0;
        int tot_grps_y = 0;

        CLManager::getGroupSize(max_warp_size, width, height, grp_size_x, grp_size_y, tot_grps_x, tot_grps_y, false);

        dout << "Work group size: " << grp_size_x << " x " << grp_size_y << endl;
        dout << "Number of groups: " << tot_grps_x << " x " << tot_grps_y << endl;

        //This array will contain the average gray value inside and outside the object

        img_sdf_half = cl::Image2D(*context, CL_MEM_READ_WRITE,
                cl::ImageFormat(CL_RGBA, CL_FLOAT), (size_t) width, (size_t) height, 0, &err);

        img_sdf_half2 = cl::Image2D(*context, CL_MEM_READ_WRITE,
                cl::ImageFormat(CL_RGBA, CL_FLOAT), (size_t) width, (size_t) height, 0, &err);

        origin = CLManager::getSizeT(0, 0, 0);
        region = CLManager::getSizeT(width, height, 1);

        vecWriteImage.push_back(evWrtImg);
        //After running this kernel we should already have the right phi
        switch (SDFmethod) {
            case SDFOZ:
                SDFoz();
                break;
            case SDFVORO:
                SDFVoro();
                break;
        }
    } catch (cl::Error ex) {
        cl->printError(ex);
    }

    delete[] array_img_out;

    return evEndSDF; // Delete is just for testing
}

/**
 * Computes the distance function of a binary image.
 * @param {int} if 1 then we compute the distance to the closest white value, if 0 then to
 * the closest black value
 */
cl::Event SignedDistFunc::voroHalfSDFBuf(int posValues) {

    //Define events used in this function
    cl::Event ev1;
    cl::Event ev2;
    cl::Event ev3;
    vector<cl::Event> vecEv1;
    vector<cl::Event> vecEv2;
    vector<cl::Event> vecEv3;

    //Creates two temporal buffers to store the temporal results.
    cl::Buffer buf_temp = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof (float) *buf_size, 0, &err);
    cl::Buffer buf_temp2 = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof (float) *buf_size, 0, &err);

    array_buf_out = new float[buf_size];

    dout << endl << "Computing HALF voro SDF" << endl;
    try {
#ifdef DEBUG
        Timer tm_step1(sdf_ts, "Step1");
        queue->finish();
        tm_step1.start(Timer::SELF);
#endif

        //----------------------  Replacing all values of cell > 0 with its index --------------------
        cl::Kernel kernelSDFVoro1(*program, "SDF_voroStep1Buf");
        kernelSDFVoro1.setArg(0, buf_mask);
        kernelSDFVoro1.setArg(1, buf_temp);
        kernelSDFVoro1.setArg(2, width);
        kernelSDFVoro1.setArg(3, height);
        kernelSDFVoro1.setArg(4, posValues); //Mode 1 is for getting SDF to values > 0
        //Mode 0 is for getting SDF to values < 0

        queue->enqueueNDRangeKernel(
                kernelSDFVoro1,
                cl::NullRange,
                cl::NDRange((size_t) width, (size_t) height),
                cl::NDRange((size_t) grp_size_x, (size_t) grp_size_y), &vecWriteImage, &ev1);

        vecEv1.push_back(ev1);

        string fileName = "";

//This section is only used for debuggin, it reads the temporal results and save them as an image
#ifdef DEBUG
        queue->finish();
        tm_step1.end();

        dout << "SDFVoro step 1 finished (all values > 0 with its index)" << endl;
        res = queue->enqueueReadBuffer(buf_temp, CL_TRUE, 0, sizeof (float) *buf_size, (void*) array_buf_out, &vecEv1, 0);

#ifdef PRINT 
        dout << "----------- Printing values of result of step1 (all values > 0 with its index) " << endl;
        MatrixUtils<float>::printImageCutted(width, height, array_buf_out, 1, cutImageW, cutImageH);
#endif

        fileName = appendStr(save_path, (char*) "temp_results/voro1IndexAtObject.png");
        ImageManager::writeGrayScaleImage((char*) fileName.c_str(), array_buf_out, FIF_PNG, width, height, 0);
#endif

        int d = 1;//Changing that we are obtaining the distance to the black values

#ifdef DEBUG
        string name = to_string( (char*)"Step2_",posValues,(char*)"");
        Timer tm_step2(sdf_ts, name.c_str());
        queue->finish();
        tm_step2.start(Timer::SELF);
#endif

        //----------------------  Obtains the closest voronoi feature for the first dimension -------------------
        cl::Kernel kernelSDFVoro2(*program, "SDF_voroStep2Buf");
        kernelSDFVoro2.setArg(0, buf_temp);
        kernelSDFVoro2.setArg(1, buf_temp2);
        kernelSDFVoro2.setArg(2, width);
        kernelSDFVoro2.setArg(3, height);
        kernelSDFVoro2.setArg(4, d);

        // Obtains closest values to the rows
        // It should do in parallel all the columns
        queue->enqueueNDRangeKernel(
                kernelSDFVoro2,
                cl::NullRange,
                cl::NDRange((size_t) width, (size_t) 1),
                cl::NDRange((size_t) grp_size_x, (size_t) 1), &vecEv1, &ev2);

        vecEv2.push_back(ev2);

#ifdef DEBUG
        queue->finish();
        tm_step2.end();

        dout << "SDFVoro step 2 finished CFV by row" << endl;
        res = queue->enqueueReadBuffer(buf_temp2, CL_TRUE, 0, sizeof (float) *buf_size, (void*) array_buf_out, &vecEv2, 0);

#ifdef PRINT 
        dout << "--------- Printing result of step 2 CFV by row" << endl;
        MatrixUtils<float>::printImageCutted(width, height, array_buf_out, 1, cutImageW, cutImageH);
#endif

        fileName = appendStr(save_path, (char*) "temp_results/voro2MaxRows.png");
        ImageManager::writeGrayScaleImage((char*) fileName.c_str(), array_buf_out, FIF_PNG, width, height, 0);
#endif

#ifdef DEBUG
        name = to_string( (char*)"Step3_",posValues,(char*)"");
        Timer tm_step3(sdf_ts, name);
        queue->finish();
        tm_step3.start(Timer::SELF);
#endif

        d = 2;

        //----------------------  Obtains the closest voronoi feature for the second dimension -------------------
        cl::Kernel kernelSDFVoro3(*program, "SDF_voroStep2Buf");
        kernelSDFVoro3.setArg(0, buf_temp2);
        if (posValues) {
            kernelSDFVoro3.setArg(1, buf_sdf_half); // The first time saves here
        } else {
            kernelSDFVoro3.setArg(1, buf_sdf_half2); // The second time saves here
        }
        kernelSDFVoro3.setArg(2, width);
        kernelSDFVoro3.setArg(3, height);
        kernelSDFVoro3.setArg(4, d);

        // Obtains the final closest distance
        queue->enqueueNDRangeKernel(
                kernelSDFVoro3,
                cl::NullRange,
                cl::NDRange((size_t) 1, (size_t) height),
                cl::NDRange((size_t) 1, (size_t) grp_size_y), &vecEv2, &ev3);

        vecEv3.push_back(ev3);

#ifdef DEBUG

        queue->finish();
        tm_step3.end();
        dout << "SDFVoro step 3 CFV overall" << endl;
        if (posValues) {
            dout << "------  Printing result of step 3 CFV overall" << endl;
            res = queue->enqueueReadBuffer(buf_sdf_half, CL_TRUE, 0, sizeof (float) *buf_size, (void*) array_buf_out, &vecEv3, 0);
        } else {
            res = queue->enqueueReadBuffer(buf_sdf_half2, CL_TRUE, 0, sizeof (float) *buf_size, (void*) array_buf_out, &vecEv3, 0);
        }

#ifdef PRINT 
        MatrixUtils<float>::printImageCutted(width, height, array_buf_out, 1, cutImageW, cutImageH);
#endif

        char* outputFile;
        if (posValues) {
            outputFile = (char*) "temp_results/VoroSecondHalf.png";
        } else {
            outputFile = (char*) "temp_results/VoroHalf.png";
        }
        fileName = appendStr(save_path, outputFile);
        ImageManager::writeGrayScaleImage((char*) fileName.c_str(), array_buf_out, FIF_PNG, width, height, 0);

        ImageManager::writeGrayScaleImage(outputFile, array_buf_out, FIF_PNG, width, height, 0);
#endif

    } catch (cl::Error ex) {
        cl->printError(ex);
    }
    return ev3;
}

cl::Event SignedDistFunc::voroHalfSDF_3DBuf(int posValues) {

    //Define events used in this function
    cl::Event ev1;// Set index at feature values
    cl::Event ev2;// Obtain partial CFP for cols dim 1
    cl::Event ev3;// Obtain partial CFP for rows dim 2
    cl::Event ev4;// Obtain partial CFP for depth dim 3
    vector<cl::Event> vecEv1;
    vector<cl::Event> vecEv2;
    vector<cl::Event> vecEv3;
    vector<cl::Event> vecEv4;

    //Creates two temporal buffers to store the temporal results.
    cl::Buffer buf_temp = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof (float) *buf_size, 0, &err);
    cl::Buffer buf_temp2 = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof (float) *buf_size, 0, &err);
    cl::Buffer buf_temp3 = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof (float) *buf_size, 0, &err);

    dout << endl << "*************** Computing HALF voro SDF for I = " << 
			posValues << "****************" << endl;
    try {
#ifdef DEBUG
        Timer tm_step1(sdf_ts, "Step1");
        queue->finish();
        tm_step1.start(Timer::SELF);
#endif

		int totalValues =  width*height*depth;
		int threadsPerGroup = max_warp_size;
		if(totalValues > threadsPerGroup){
			//Making sure that totalValues is divisible by threadsPerGroup 
			while( totalValues % threadsPerGroup != 0) {
				threadsPerGroup--;
			}
		}else{
			threadsPerGroup=totalValues;
		}

		dout << "Total number of values: " << totalValues << endl;
		dout << "Number of threads per group: " << threadsPerGroup << endl;

        queue->finish();//Not neccessary, has already been called
        //----------------------  Replacing all values of cell > 0 with its index --------------------
        cl::Kernel kernelSDFVoro1(*program, "SDF_voroStep1Buf");
        kernelSDFVoro1.setArg(0, buf_mask);
        kernelSDFVoro1.setArg(1, buf_temp);
        kernelSDFVoro1.setArg(2, totalValues);
        kernelSDFVoro1.setArg(3, posValues); //Mode 1 is for getting SDF to values > 0
        //Mode 0 is for getting SDF to values < 0

        queue->enqueueNDRangeKernel(
                kernelSDFVoro1,
                cl::NullRange,
                cl::NDRange((size_t) totalValues),
                cl::NDRange((size_t) threadsPerGroup), &vecWriteImage, &ev1);

        vecEv1.push_back(ev1);

//This section is only used for debuggin, it reads the temporal results and save them as an image
#ifdef DEBUG
        queue->finish();
        tm_step1.end();
		
        dout << "SDFVoro step 1 finished (all values > 0 with its index)" << endl;
        res = queue->enqueueReadBuffer(buf_temp, CL_TRUE, 0, sizeof (float) *buf_size, 
				(void*) array_buf_out, &vecEv1, 0);

        queue->finish();
        dout << "----------- Printing values of result of step1 (all values > 0 with its index) " << endl;
#ifdef PRINT
        MatrixUtils<float>::print3DImage(width, height, depth,  array_buf_out);
#endif
#endif

		// --------------------------- Running by cols ---------------------
        int dimension = 1;

		ev2  = runStep2(buf_temp, buf_temp2, width, height, depth,
				dimension, vecEv1 , posValues);

        vecEv2.push_back(ev2);

		// --------------------------- Running by rows ---------------------
        dimension = 2;
		ev3  = runStep2(buf_temp2, buf_temp3, width, height, depth,
				dimension, vecEv2 , posValues);

        vecEv3.push_back(ev3);

		// --------------------------- Running by z ---------------------

		std::cout << std::fixed;
		std::cout << std::setprecision(2);
        dimension = 3;
        if (posValues) {
			ev4  = runStep2(buf_temp3, buf_sdf_half, width, height, depth,
					dimension, vecEv3 , posValues);
        } else {
			ev4  = runStep2(buf_temp3, buf_sdf_half2, width, height, depth,
					dimension, vecEv3 , posValues);
        }

        vecEv4.push_back(ev4);

    } catch (cl::Error ex) {
        cl->printError(ex);
    }
    return ev4;
}

cl::Event SignedDistFunc::runStep2(cl::Buffer& inputBuffer, cl::Buffer& outputBuffer,
		int w, int h, int z, int dim, vector<cl::Event>& vecEvPrev, int posValues){

		cl::Event event;

#ifdef DEBUG
        string name = to_string( (char*)"Step",(dim+1),(char*)"_") ;
        Timer tm_stepn(sdf_ts, name.c_str());
        queue->finish();
        tm_stepn.start(Timer::SELF);
#endif

        //----------------------  Obtains the closest voronoi feature for the first dimension -------------------
        cl::Kernel kernelSDFVoro(*program, "SDF_voroStep2Buf");
        kernelSDFVoro.setArg(0, inputBuffer);
        kernelSDFVoro.setArg(1, outputBuffer);
        kernelSDFVoro.setArg(2, w);
        kernelSDFVoro.setArg(3, h);
        kernelSDFVoro.setArg(4, z);
        kernelSDFVoro.setArg(5, dim);

        int tot_grps_x = 0;
        int tot_grps_y = 0;

		switch(dim){
			case 1:
				CLManager::getGroupSize(max_warp_size, width, depth, 
						grp_size_x, grp_size_y, tot_grps_x, tot_grps_y, true);

				queue->enqueueNDRangeKernel(
						kernelSDFVoro,
						cl::NullRange,
						cl::NDRange((size_t) width, (size_t) 1, (size_t) depth) ,
						cl::NDRange((size_t) grp_size_x, (size_t) 1, (size_t) grp_size_y), &vecEvPrev, &event);
				break;
			case 2:
				CLManager::getGroupSize(max_warp_size, height, depth, 
						grp_size_x, grp_size_y, tot_grps_x, tot_grps_y, true);

				queue->enqueueNDRangeKernel(
						kernelSDFVoro,
						cl::NullRange,
						cl::NDRange((size_t) 1, (size_t) height, (size_t) depth) ,
						cl::NDRange((size_t) 1, (size_t) grp_size_x,(size_t)  grp_size_y), &vecEvPrev, &event);
				break;
			case 3:
				CLManager::getGroupSize(max_warp_size, width, height, 
						grp_size_x, grp_size_y, tot_grps_x, tot_grps_y, true);

				queue->enqueueNDRangeKernel(
						kernelSDFVoro,
						cl::NullRange,
						cl::NDRange((size_t) width, (size_t) height, (size_t) 1) ,
						cl::NDRange((size_t) grp_size_x, (size_t) grp_size_y, (size_t) 1), &vecEvPrev, &event);
				break;

		}

#ifdef DEBUG

		vector<cl::Event> vecEvPrint;
        vecEvPrint.push_back(event);

        queue->finish();
        tm_stepn.end();

        dout << "SDFVoro step 2 finished CFV for dimension:  " << dim << endl;
        res = queue->enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof (float) *buf_size, (void*) array_buf_out, &vecEvPrint, 0);

        queue->finish();
        dout << "----------- Printing values of result of step" << dim << endl;
#ifdef PRINT
        MatrixUtils<float>::print3DImage(width, height, depth,  array_buf_out);
#endif
#endif

		return event;
}

cl::Event SignedDistFunc::voroHalfSDF(int posValues) {

    cl::Event ev1;
    cl::Event ev2;
    cl::Event ev3;
    vector<cl::Event> vecEv1;
    vector<cl::Event> vecEv2;
    vector<cl::Event> vecEv3;

    cl::Image2D img_temp = cl::Image2D(*context, CL_MEM_READ_WRITE,
            cl::ImageFormat(CL_RGBA, CL_FLOAT), (size_t) width, (size_t) height, 0, &err);

    cl::Image2D img_temp2 = cl::Image2D(*context, CL_MEM_READ_WRITE,
            cl::ImageFormat(CL_RGBA, CL_FLOAT), (size_t) width, (size_t) height, 0, &err);

    dout << endl << "Computing HALF voro SDF" << endl;
    float * array_img_out = new float[width * height * 4];
    try {

        cl::Sampler sampler(*context, CL_FALSE, CL_ADDRESS_REPEAT, CL_FILTER_NEAREST, &err);

        cl::Kernel kernelSDFVoro1(*program, "SDF_voroStep1");
        kernelSDFVoro1.setArg(0, img_mask);
        kernelSDFVoro1.setArg(1, img_temp);
        kernelSDFVoro1.setArg(2, width);
        kernelSDFVoro1.setArg(3, height);
        kernelSDFVoro1.setArg(4, sampler);
        kernelSDFVoro1.setArg(5, posValues); //Mode 1 is for getting SDF to values > 0
        //Mode 0 is for getting SDF to values < 0

        // Replaces all values of cell > 0 with its index
        queue->enqueueNDRangeKernel(
                kernelSDFVoro1,
                cl::NullRange,
                cl::NDRange((size_t) width, (size_t) height),
                cl::NDRange((size_t) grp_size_x, (size_t) grp_size_y), &vecWriteImage, &ev1);

        vecEv1.push_back(ev1);

#ifdef DEBUG
        dout << "SDFVoro step 1 finished (all values > 0 with its index)" << endl;
        res = queue->enqueueReadImage(img_temp,
                CL_TRUE, origin, region, 0, 0, (void*) array_img_out, &vecEv1, 0);

#ifdef PRINT 
        MatrixUtils<float>::printImageCutted(width, height, array_img_out, 4, cutImageW, cutImageH);
#endif 
        ImageManager::writeImage((char*) "temp_results/voro1IndexAtObject.png",
                array_img_out, FIF_PNG, width, height);

#endif 

        int d = 1;

        cl::Kernel kernelSDFVoro2(*program, "SDF_voroStep2");
        kernelSDFVoro2.setArg(0, img_temp);
        kernelSDFVoro2.setArg(1, img_temp2);
        kernelSDFVoro2.setArg(2, width);
        kernelSDFVoro2.setArg(3, height);
        kernelSDFVoro2.setArg(4, sampler);
        kernelSDFVoro2.setArg(5, d);

        // Obtains closest values to the rows
        // It should do in parallel all the columns
        queue->enqueueNDRangeKernel(
                kernelSDFVoro2,
                cl::NullRange,
                cl::NDRange((size_t) width, (size_t) 1),
                cl::NDRange((size_t) grp_size_x, (size_t) 1), &vecEv1, &ev2);

        vecEv2.push_back(ev2);

#ifdef DEBUG
        dout << "SDFVoro step 2 finished CFV by row" << endl;
        res = queue->enqueueReadImage(img_temp2,
                CL_TRUE, origin, region, 0, 0, (void*) array_img_out, &vecEv2, 0);

#ifdef PRINT 
        MatrixUtils<float>::printImageCutted(width, height, array_img_out, 4, cutImageW, cutImageH);
#endif 

        ImageManager::writeImage((char*) "temp_results/voro2MaxRows.png",
                array_img_out, FIF_PNG, width, height);
#endif 

        //		queue->enqueueCopyImage(img_write, img_read, origin, origin, region);

        d = 2;
        cl::Kernel kernelSDFVoro3(*program, "SDF_voroStep2");
        kernelSDFVoro3.setArg(0, img_temp2);
        if (posValues) {
            kernelSDFVoro3.setArg(1, img_sdf_half); // The first time saves here
        } else {
            kernelSDFVoro3.setArg(1, img_sdf_half2); // The second time saves here
        }
        kernelSDFVoro3.setArg(2, width);
        kernelSDFVoro3.setArg(3, height);
        kernelSDFVoro3.setArg(4, sampler);
        kernelSDFVoro3.setArg(5, d);

        // Obtains the final closest distance
        queue->enqueueNDRangeKernel(
                kernelSDFVoro3,
                cl::NullRange,
                cl::NDRange((size_t) 1, (size_t) height),
                cl::NDRange((size_t) 1, (size_t) grp_size_y), &vecEv2, &ev3);

        vecEv3.push_back(ev3);

#ifdef DEBUG
        dout << "SDFVoro step 3 CFV overall" << endl;
        if (posValues) {
            res = queue->enqueueReadImage(img_sdf_half,
                    CL_TRUE, origin, region, 0, 0, (void*) array_img_out, &vecEv3, 0);
        } else {
            res = queue->enqueueReadImage(img_sdf_half2,
                    CL_TRUE, origin, region, 0, 0, (void*) array_img_out, &vecEv3, 0);

        }

#ifdef PRINT 
        MatrixUtils<float>::printImageCutted(width, height, array_img_out, 4, cutImageW, cutImageH);
#endif 

        if (posValues) {
            ImageManager::writeImage((char*) "temp_results/VoroHalf.png",
                    array_img_out, FIF_PNG, width, height);
        } else {
            ImageManager::writeImage((char*) "temp_results/VoroSecondHalf.png",
                    array_img_out, FIF_PNG, width, height);
        }
#endif 

    } catch (cl::Error ex) {
        cl->printError(ex);
    }
    return ev3;
}

cl::Event SignedDistFunc::SDFVoro() {

    cl::Sampler sampler(*context, CL_FALSE, CL_ADDRESS_REPEAT, CL_FILTER_NEAREST, &err);

    try {
        cl::Event lastPosEvent = voroHalfSDF(1);
        cl::Event lastNegEvent = voroHalfSDF(0);

        dout << endl << "Merging PHIs" << endl;

        cl::Kernel kernelMergePhis(*program, "mergePhis");
        kernelMergePhis.setArg(0, img_sdf_half);
        kernelMergePhis.setArg(1, img_sdf_half2);
        kernelMergePhis.setArg(2, img_sdf);
        kernelMergePhis.setArg(3, sampler);

        vector<cl::Event> prevEvents;
        prevEvents.push_back(lastNegEvent);
        prevEvents.push_back(lastPosEvent);

        queue->enqueueNDRangeKernel(
                kernelMergePhis,
                cl::NullRange,
                cl::NDRange((size_t) width, (size_t) height),
                cl::NDRange((size_t) grp_size_x, (size_t) grp_size_y), &prevEvents, &evEndSDF);

        vector<cl::Event> vecEvMerg;
        vecEvMerg.push_back(evEndSDF);

    } catch (cl::Error ex) {
        cl->printError(ex);
    }
    return evEndSDF;
}

/**
 * Computes the SDF function using the Voronoi (WHICH) method.
 * It computes the SDF for the binary image obtained from the mas and
 * then it does it again by changing the colors of the binary image
 */
cl::Event SignedDistFunc::SDFVoroBuf() {

    try {
        // Computes the distance from all pixels to the closest pixel > 0
        cl::Event lastPosEvent = voroHalfSDFBuf(1);
        // Computes the distance from all pixels to the closest pixel == 0
        cl::Event lastNegEvent = voroHalfSDFBuf(0);


#ifdef DEBUG
        Timer tm_merge(sdf_ts, "Merging");
        queue->finish();
        tm_merge.start(Timer::SELF);
#endif

        dout << "Merging PHIs........." << endl;
        dout << width << " x " << height << endl;
        dout << "Work group size: " << grp_size_x << " x " << grp_size_y << endl;

        // Merges the two buffers sdf_half and sdf_half2 into the final SDF with 
        // negative distances for pixels > 0
        kernelMergePhis = cl::Kernel(*program, "mergePhisBuf");
        kernelMergePhis.setArg(0, buf_sdf_half);
        kernelMergePhis.setArg(1, buf_sdf_half2);
        kernelMergePhis.setArg(2, buf_sdf);
        kernelMergePhis.setArg(3, width);
        kernelMergePhis.setArg(4, height);

        vector<cl::Event> prevEvents;
        prevEvents.push_back(lastNegEvent);
        prevEvents.push_back(lastPosEvent);

        queue->enqueueNDRangeKernel(
                kernelMergePhis,
                cl::NullRange,
                cl::NDRange((size_t) width, (size_t) height),
                cl::NDRange((size_t) grp_size_x, (size_t) grp_size_y), &prevEvents, &evEndSDF);

        vector<cl::Event> vecEvMerg;
        vecEvMerg.push_back(evEndSDF);

#ifdef DEBUG
        tm_merge.end();
#endif

#ifdef PRINT 
        dout << " SDF result: " << endl;
        res = queue->enqueueReadBuffer(buf_sdf, CL_TRUE, 0, sizeof (float) *buf_size, (void*) array_buf_out, &vecEvMerg, 0);
        MatrixUtils<float>::printImageCutted(width,height,array_buf_out,1, cutImageW, cutImageH);
#endif 

    } catch (cl::Error ex) {
        cl->printError(ex);
    }
    return evEndSDF;
}

/**
 * Computes the SDF function using the Voronoi (WHICH) method.
 * It computes the SDF for the binary image obtained from the mas and
 * then it does it again by changing the colors of the binary image
 */
cl::Event SignedDistFunc::SDF3DVoroBuf() {

    try {
        // Computes the distance from all pixels to the closest pixel > 0
        cl::Event lastPosEvent = voroHalfSDF_3DBuf(1);

        // Computes the distance from all pixels to the closest pixel == 0
        cl::Event lastNegEvent = voroHalfSDF_3DBuf(0);

#ifdef DEBUG
        Timer tm_merge(sdf_ts, "Merging");
        queue->finish();
        tm_merge.start(Timer::SELF);
#endif

		int tot_grps_x, tot_grps_y; 
        dout << "Merging PHIs........." << endl;
		CLManager::getGroupSize(max_warp_size, width, height, 
				grp_size_x, grp_size_y, tot_grps_x, tot_grps_y, true);

        dout << width << " x " << height << endl;
        dout << "Work group size: " << grp_size_x << " x " << grp_size_y << endl;

        // Merges the two buffers sdf_half and sdf_half2 into the final SDF with 
        // negative distances for pixels > 0
        kernelMergePhis = cl::Kernel(*program, "mergePhisBuf");
        kernelMergePhis.setArg(0, buf_sdf_half);
        kernelMergePhis.setArg(1, buf_sdf_half2);
        kernelMergePhis.setArg(2, buf_sdf);
        kernelMergePhis.setArg(3, width);
        kernelMergePhis.setArg(4, height);
        kernelMergePhis.setArg(5, depth);

        vector<cl::Event> prevEvents;
        prevEvents.push_back(lastNegEvent);
        prevEvents.push_back(lastPosEvent);

        queue->enqueueNDRangeKernel(
                kernelMergePhis,
                cl::NullRange,
                cl::NDRange((size_t) width, (size_t) height),
                cl::NDRange((size_t) grp_size_x, (size_t) grp_size_y), &prevEvents, &evEndSDF);

        vector<cl::Event> vecEvMerg;
        vecEvMerg.push_back(evEndSDF);

#ifdef DEBUG
        tm_merge.end();

        queue->finish();

		int buf_size = width*height*depth;
        res = queue->enqueueReadBuffer(buf_sdf, CL_TRUE, 0, sizeof (float) *buf_size, (void*) array_buf_out, &vecEvMerg, 0);

        queue->finish();
        dout << "----------- Printing SDF yeah!! ------------" << endl;
#ifdef PRINT
        MatrixUtils<float>::print3DImage(width, height, depth,  array_buf_out);
#endif
#endif

    } catch (cl::Error ex) {
        cl->printError(ex);
    }
    return evEndSDF;
}

int SignedDistFunc::SDFoz() {

    /*
       try {
       origin = ImageManager::getSizeT(0, 0, 0);
       region = ImageManager::getSizeT(width, height, 1);

       cl::Sampler sampler(context, CL_TRUE,
       CL_ADDRESS_REPEAT, CL_FILTER_LINEAR, &err);

       queue->finish();

       cl::Kernel kernelSDFOZ(program, "SDFOZ");
       kernelSDFOZ.setArg(0, img_mask);
       kernelSDFOZ.setArg(1, img_sdf);
       kernelSDFOZ.setArg(2, width);
       kernelSDFOZ.setArg(3, height);
       kernelSDFOZ.setArg(4, sampler);

    // Do the work
    queue->enqueueNDRangeKernel(
    kernelSDFOZ,
    cl::NullRange,
    cl::NDRange((size_t) width, (size_t) height),
    cl::NDRange((size_t) grp_size_x, (size_t) grp_size_y));

    } catch (cl::Error ex) {
    cl.printError(ex);

    return EXIT_FAILURE;
    }
    */
    return EXIT_SUCCESS;
}

/**
 * This is the main function in charge of computing the 3D Signed Distance function of a 
 * 3D binary mask
 * @param clOrig A CLManager object with an initialized CL context, queue and device selected
 * @param buf_mask Initial binary mask where the SDF will be computed 
 * @param buf_sdf  Buffer where the SDF will be set as an output 
 * @param max_warp_size 
 * @param width
 * @param height
 * @param depth
 * @param evWrtImg It has the event that is required to wait for writing the buffers
 * @param save_path
 * @return 
 */
cl::Event SignedDistFunc::run3DSDFBuf(CLManager* clOrig,  cl::Buffer& buf_mask, cl::Buffer& buf_sdf, 
		int max_warp_size, int width, int height, int depth, cl::Event evWrtImg, char* save_path) {
	
#ifdef DEBUG
    Timer tm_prev(sdf_ts, "Prev");
    tm_prev.start(Timer::SELF);
#endif
	
    this->save_path = save_path;
    this->buf_mask = buf_mask;
    this->buf_sdf = buf_sdf;
    cl = clOrig;
	
    buf_size = width*height*depth;
	
    try {
		
        context = cl->getContext();
        queue = cl->getQueue();
        program = cl->getProgram();
		
        array_buf_out = new float[buf_size];
		
        this->max_warp_size = max_warp_size;
		
        this->width = width;
        this->height = height;
        this->depth = depth;
		
        origin = CLManager::getSizeT(0, 0, 0);
        region = CLManager::getSizeT(width, height, depth);
		
		//These buffers will have the SDF to closest 0 value and to clostest >0 value
        buf_sdf_half = cl::Buffer(*context, CL_MEM_READ_WRITE, buf_size * sizeof (float), NULL, &err);
        buf_sdf_half2 = cl::Buffer(*context, CL_MEM_READ_WRITE, buf_size * sizeof (float), NULL, &err);
		
#ifdef DEBUG
        queue->finish();
        tm_prev.end();
#endif
        vecWriteImage.push_back(evWrtImg);

		evEndSDF = SDF3DVoroBuf();
#ifdef DEBUG
        sdf_ts.dumpTimings();
#endif
    } catch (cl::Error ex) {
        cl->printError(ex);
    }
	
    delete[] array_buf_out;
	
    return evEndSDF; // Delete is just for testing
}
