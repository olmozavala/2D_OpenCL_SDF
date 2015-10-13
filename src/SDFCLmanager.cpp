/* 
 * File:   SDFCLmanager.cpp
 * Author: olmozavala
 * 
 * Created on November 24, 2011, 9:06 PM
 */

#include "CLManager/CLManager.h"
#include "SDFCLmanager.h"
#include "SignedDistFunc.h"
#include <sstream>
#include "CL/cl.hpp"
#include "debug.h"
#include "MatrixUtils/MatrixUtils.h"

#define MAXF 1
#define MAXDPHIDT 2
#define EPS .00001

// These two variables define how many pixels display as a matrix
// when debugging and printing values
#define cutImageW 12
#define cutImageH 16

SDFCLmanager::SDFCLmanager() {
}

SDFCLmanager::SDFCLmanager(const SDFCLmanager& orig) {
}

SDFCLmanager::~SDFCLmanager() {
}

/**
 * Main method. It computes the SDF of a binary image.
 * @param {int} Method (0 = Searching radially, 1 = Using Voronoi Maurer method)
 * @param {char*} Input name for binary image
 * @param {char*} Output name of SDF as image
 * @return int EXIT_SUCCESS or EXIT_FAILURE
 */
int SDFCLmanager::runBuf(int SDFmethod, char* inputFile, char* outputFile) {

    //Create timers
    Timings ts;
    Timer tm_context(ts, "ContextAndLoad");
    Timer tm_sdf(ts, "SDF_kernels");
    Timer tm_all(ts, "All_time");

    int width;
    int height;

    int* arr_buf_mask;
    float* array_buf_out;

    tm_all.start(Timer::SELF); // Start time for ALLL

    //Loads the input image and the values for width and height
    arr_buf_mask = ImageManager::loadImageGrayScale(inputFile, width, height);

    tm_context.start(Timer::SELF);//Start time for context loading
    try {

        // Create the program from source
        cl.initContext(false); //false = not using OpenGL
        switch (SDFmethod) {
            case SDFOZ:
                cl.addSource((char*) "src/resources/SDF.cl");
                break;
            case SDFVORO:
                cl.addSource((char*) "src/resources/SDFVoroBuf.cl");
                break;
        }

        // Initialize OpenCL queue, context and program
        cl.initQueue();
        context = cl.getContext();
        queue = cl.getQueue();
        program = cl.getProgram();

		#ifdef PRINT 
            //If Deep Debuging then print the values of the image
            MatrixUtils<int>::printImageCutted(width, height, arr_buf_mask, 1, cutImageW, cutImageH);
		#endif

        int max_warp_size = 16; 
        max_warp_size = cl.getMaxWorkGroupSize(0);


		#ifdef DEBUG
			cout << "---------Device info -------------" << endl;
			cl.getDeviceInfo(0);
		#endif

		dout << endl << "Image size: " << width << " x " << height << endl;

        int buf_size = width*height;
        array_buf_out = new float[buf_size];

        //Creates two buffers, one for the input binary image and one for the SDF
        cl::Buffer buf_mask = cl::Buffer(*context, CL_MEM_READ_WRITE, buf_size * sizeof (unsigned int), NULL, &err);
        cl::Buffer buf_sdf = cl::Buffer(*context, CL_MEM_WRITE_ONLY, buf_size * sizeof (float), NULL, &err);

        cl::Event evWrtImg;
        dout << "Writing image..." << endl;
        //Writes the input binary image
        err = queue->enqueueWriteBuffer(buf_mask, CL_TRUE, 0, sizeof (unsigned char) *buf_size, (void*) arr_buf_mask, 0, &evWrtImg);

        cl::Event evSDF;

        dout << "Running .... " << endl << endl;
        SignedDistFunc sdfObj;

        queue->finish();
        tm_context.end(); //Finish timer for context creation

        //for(int i=0; i<10; i++){
            tm_sdf.start(Timer::SELF);//Starts time for kernels
            evSDF = sdfObj.runSDFBuf(&cl, SDFmethod, buf_mask, buf_sdf, max_warp_size, width, height, evWrtImg, (char*) "./");
            queue->finish();
            tm_sdf.end();//Ends time for kernels
        //}

        vector<cl::Event> vecFinishSDF;
        vecFinishSDF.push_back(evSDF);

        dout << "Writing Signed Distance Function" << endl;
        res = queue->enqueueReadBuffer(buf_sdf, CL_TRUE, 0, sizeof (float) *buf_size, (void*) array_buf_out, &vecFinishSDF, 0);

#ifdef PRINT 
        dout << "----------- Printing values of result of step1 " << endl;
        MatrixUtils<float>::printImageCutted(width, height, array_buf_out, 1, cutImageW, cutImageH);
#endif

        ImageManager::writeGrayScaleImage((char*) outputFile, array_buf_out, FIF_PNG, width, height, 0);

        dout << "SUCESS!!!" << endl;
        tm_all.end();
        ts.dumpTimings();
    } catch (cl::Error ex) {
        cl.printError(ex);

        return EXIT_FAILURE;
    }

    delete[] arr_buf_mask;
    delete[] array_buf_out;

    return EXIT_SUCCESS;
}

int SDFCLmanager::run(int SDFmethod, char* inputFile, char* outputFile) {

    CLManager cl;

    cl::Context* context;
    cl::CommandQueue* queue;
    cl::Program* program;

    int width;
    int height;

    cl::size_t < 3 > origin;
    cl::size_t < 3 > region;

    int* arr_img_mask;
    float* array_img_out;

    try {
        // Create the program from source
        cl.initContext(false);
        switch (SDFmethod) {
            case SDFOZ:
                cl.addSource((char*) "src/resources/SDF.cl");
                break;
            case SDFVORO:
                cl.addSource((char*) "src/resources/SDFVoro.cl");
                break;
        }

        cl.initQueue();

        context = cl.getContext();
        queue = cl.getQueue();
        program = cl.getProgram();

        arr_img_mask = ImageManager::loadImage(inputFile, width, height);

        origin = CLManager::getSizeT(0, 0, 0);
        region = CLManager::getSizeT(width, height, 1);

        //        cout << "---------Device info -------------" << endlmo
        cl.getDeviceInfo(0);

        int max_warp_size = 0; 
        max_warp_size = cl.getMaxWorkGroupSize(0);

        dout << endl << "Image size: " << width << " x " << height << endl;
        array_img_out = new float[width * height * 4];

        cl::Image2D img_mask = cl::Image2D(*context, CL_MEM_READ_WRITE,
                cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8), (size_t) width, (size_t) height, 0, &err);

        cl::Image2D img_sdf = cl::Image2D(*context, CL_MEM_WRITE_ONLY,
                cl::ImageFormat(CL_RGBA, CL_FLOAT), (size_t) width, (size_t) height, 0, &err);

        cl::Event evWrtImg;
        dout << "Writing image..." << endl;
        err = queue->enqueueWriteImage(img_mask, CL_FALSE, origin, region, 0, 0, (void*) arr_img_mask, 0, &evWrtImg);

        cl::Event evSDF;
        dout << "Running .... " << endl << endl;
        SignedDistFunc sdfObj;

        evSDF = sdfObj.runSDF(&cl, SDFmethod, img_mask, img_sdf, max_warp_size, width, height, evWrtImg, (char*) "images");

        vector<cl::Event> vecFinishSDF;
        vecFinishSDF.push_back(evSDF);

        dout << "Reading resulting SDF... " << endl;
        res = queue->enqueueReadImage(img_sdf,
                CL_TRUE, origin, region, 0, 0, (void*) array_img_out, &vecFinishSDF, 0);

        dout << "After reading final image" << endl;
        ImageManager::writeImage(outputFile,
                array_img_out, FIF_PNG, width, height);

        dout << "SUCESS!!!" << endl;
    } catch (cl::Error ex) {
        cl.printError(ex);

        return EXIT_FAILURE;
    }
    delete[] arr_img_mask;
    delete[] array_img_out;

    return EXIT_SUCCESS;
}

/**
 * This is the main function used to compute a 3D SDF using gif images as initial masks
 * @param inputFile	Name of the input file
 * @param outputFolder Folder where to put the SDF result as a set of images
 * @return 
 */
int SDFCLmanager::run3dBuf(char* inputFile, char* outputFolder) {

    //Create timers
    Timings ts;
    Timer tm_context(ts, "ContextAndLoad");
    Timer tm_sdf(ts, "SDF_kernels");
    Timer tm_all(ts, "All_time");

    int width, height, depth;

    unsigned char* arr_buf_mask;
    float* array_buf_out;

    tm_all.start(Timer::SELF); // Start time for ALLL

    //Loads the input image and the values for width and height
    arr_buf_mask = ImageManager::load3dImageGif(inputFile, width, height, depth);

	//Image has been loaded correctly
//	MatrixUtils<unsigned char>::print3DImage(width, height, depth,  arr_buf_mask);

    tm_context.start(Timer::SELF);//Start time for context loading
    try {

        // Create the program from source
        cl.initContext(false); //false = not using OpenGL
		cl.addSource((char*) "src/resources/SDFVoroBuf3D.cl");

        // Initialize OpenCL queue, context and program
        cl.initQueue();
        context = cl.getContext();
        queue = cl.getQueue();
        program = cl.getProgram();

        int max_warp_size = 0; 
        max_warp_size = cl.getMaxWorkGroupSize(0);
        max_warp_size = 512;

		#ifdef DEBUG
			dout << "---------Device info -------------" << endl;
			cl.getDeviceInfo(0);
		#endif

		dout << endl << "Image size: " << width << " x " << height << " x " << depth << endl;

        int buf_size = width*height*depth;
        array_buf_out = new float[buf_size];

        //Creates two buffers, one for the input binary image and one for the SDF
        cl::Buffer buf_mask = cl::Buffer(*context, CL_MEM_READ_WRITE, buf_size * sizeof (unsigned char), NULL, &err);
        cl::Buffer buf_sdf = cl::Buffer(*context, CL_MEM_WRITE_ONLY, buf_size * sizeof (float), NULL, &err);

        cl::Event evWrtImg;
        dout << "Writing image..." << endl;
        //Writes the input binary image into the buffer
        err = queue->enqueueWriteBuffer(buf_mask, CL_TRUE, 0, sizeof (unsigned char) *buf_size, (void*) arr_buf_mask, 0, &evWrtImg);

		
        cl::Event evSDF;

        dout << "Running .... " << endl << endl;
        SignedDistFunc sdfObj;

        tm_context.end(); //Finish timer for context creation
        queue->finish();// Wait for the image to be written
		
		tm_sdf.start(Timer::SELF);//Starts time for kernels

		evSDF = sdfObj.run3DSDFBuf(&cl, buf_mask, buf_sdf, max_warp_size, width, height, depth, evWrtImg, outputFolder);
		
		queue->finish();
		tm_sdf.end();//Ends time for kernels
		
        vector<cl::Event> vecFinishSDF;
        vecFinishSDF.push_back(evSDF);
		
        dout << "Writing Signed Distance Function" << endl;
        res = queue->enqueueReadBuffer(buf_sdf, CL_TRUE, 0, sizeof (float) *buf_size, (void*) array_buf_out, &vecFinishSDF, 0);
		
		dout << "Writing images at: " << outputFolder << endl;
        ImageManager::write3DImage((char*) outputFolder, array_buf_out, width, height, depth, 0);
		
        dout << "SUCESS!!!" << endl;
        tm_all.end();
        ts.dumpTimings();
    } catch (cl::Error ex) {
        cl.printError(ex);
		
        return EXIT_FAILURE;
    }
	
    delete[] arr_buf_mask;
    delete[] array_buf_out;
	
    return EXIT_SUCCESS;
}
