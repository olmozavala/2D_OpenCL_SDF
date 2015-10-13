
#define __CL_ENABLE_EXCEPTIONS

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include "FreeImage.h"
#include "SignedDistFunc.h"
#include "SDFCLmanager.h"

using namespace std;

inline std::string appendStr(char* prev, char* post) {
    std::stringstream ss;
    ss << prev << post;
    return ss.str();
}

/**
 * This function simple selects examples from the image folder
 */
void selectExample(int example, char** inputImage, char** outputImage){

    switch (example) {
        case 0:
            //*inputImage = (char*) "images/Mini.png";
            //*outputImage = (char*) "images/MiniResult.png";
            *inputImage = (char*) "images/Circle.png";
            *outputImage = (char*) "images/Results/CircleResult.jpg";
            break;
        case 1:
            *inputImage = (char*) "images/Basic.png";
            *outputImage = (char*) "images/Results/BasicResult.png";
            break;
        case 2:
            *inputImage = (char*) "images/Small.png";
            *outputImage = (char*) "images/Results/SmallResult.png";
            break;
        case 3:
            *inputImage = (char*) "images/Medium.png";
            *outputImage = (char*) "images/Results/MediumResult.png";
            break;
        case 4:
            *inputImage = (char*) "images/MediumBig.png";
            *outputImage = (char*) "images/Results/MediumBigResult.png";
            break;
        case 5:
            *inputImage = (char*) "images/BigRealTest.png";
            *outputImage = (char*) "images/Results/BigTestResult.png";
            break;
        case 6:
            *inputImage = (char*) "images/BigNew.png";
            *outputImage = (char*) "images/Results/BigNewTest.png";
            break;
    }

}

/**
 * This is the main function, it simply runs 
 * the SDF with an specific example or the one received as parameter
 */
int main(int argc, char** args){

    int example = 0; //Example to use, by default is 1
    if(argc < 2){
        cout<<"Please select an example from 1 to 6. Currently using "<< example <<" as default" << endl;
    }else{
        example= atoi(args[1]);
    }

    char* inputImage;
    char* outputImage;

    //Selects the example we want to use
    selectExample(example, &inputImage, &outputImage);

    SDFCLmanager ac = SDFCLmanager();
    return ac.runBuf(SDFVORO, (char*) inputImage, (char*) outputImage); //With buffers

    /* //This part of the code is only used for my PhD performance tests
       string save_path = "images/TestSizes/";
       string result_path = "images/TestSizes/result";
    //string files[16] = { "256.png", "512.png", "768.png", "1024.png", "1280.png", "1536.png", "1792.png", "2048.png", "2304.png", "2560.png", "2816.png", "3072.png" ,"3328.png" ,"3584.png" ,"3840.png" ,"4096.png" }; 
    string files[4] = { "3328.png" ,"3584.png" ,"3840.png" ,"4096.png" };

    for (int i = 0; i < 4; i++) {
    string inputImage = appendStr((char*) save_path.c_str(), (char*) files[i].c_str());
    string outputImage = appendStr((char*) result_path.c_str(), (char*) files[i].c_str());
    int result = ac.runBuf(SDFVORO, (char*) inputImage.c_str(), (char*) outputImage.c_str()); //With buffers
    if (result != EXIT_SUCCESS) {
    return result;
    }
    }
    */
}
