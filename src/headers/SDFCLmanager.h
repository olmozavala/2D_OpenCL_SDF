/* 
 * File:   SDFCLmanager.h
 * Author: olmozavala
 *
 * Created on November 24, 2011, 9:06 PM
 */

#ifndef SDFCLMANAGER_H
#define	SDFCLMANAGER_H

#include <CL/cl.hpp>
#include <sstream>

class SDFCLmanager {
public:
    SDFCLmanager();
    SDFCLmanager(const SDFCLmanager& orig);
    virtual ~SDFCLmanager();
    int run(int SDFmethod, char* inputFile, char* outputFile);
    int runBuf(int SDFmethod, char* inputFile, char* outputFile);
    int run3dBuf( char* inputFile, char* outputFile);
private:
		
    cl_int err;
    cl_int res;

    //This is a helper class for OpenCL
    CLManager cl;

    cl::Context* context;
    cl::CommandQueue* queue;
    cl::Program* program;


};

#endif	/* SDFCLMANAGER_H */

