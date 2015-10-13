__kernel void
SDFOZ(read_only image2d_t src, write_only image2d_t dst,
        int width, int height, sampler_t sampler) {
    int col = (int)get_global_id(0);
    int row = (int)get_global_id(1);

	if( (row > height) || (col > width))
		return;

    float ctrx = (float)col/width;
    float ctry = (float)row/height;

//    float angle_step = 2*M_PI_F/(pow((float)2,2));
    int totalAngles = 4;
    float angle_step = (float)2*M_PI_F/(float)totalAngles;
    bool notfound = true;

    float mag = 1/(max(width,height)/2);//Initial magnitude
    float magstep = .01;//approx one pixel size

    float newx = 0;
    float newy = 0;
    float4 newval = (float4)(0,0,0,255);

    int iter = 1;

	float maxMag = 1.5;

    uint4 gx = read_imageui(src, sampler, (float2)(ctrx,ctry));
    if(gx.x<1){
        while(notfound){
    //        angle_step = 2*M_PI_F/((float)pow((float)(iter+1),2));
			if(totalAngles < 500)
				totalAngles *= 2;

            angle_step = (float)2*M_PI_F/(float)totalAngles;
            for(float i=0; i<2*M_PI_F; i+=angle_step){
                newx = ctrx + sin(i)*mag;
                newy = ctry + cos(i)*mag;
                if( newx >= 1 || newy >= 1 || newx < 0 || newy < 0 ){
                    continue;
                }
                else{
                    gx = read_imageui(src, sampler, (float2)(newx,newy));
                    if(gx.x >= 1){
                        newval.y = mag*255;
						newval.x = mag*sqrt( (float)width*width + height*height);
                        notfound = false;
                        break;
                    }
                }
            }
            mag = mag + magstep;
            if(mag > maxMag){
                newval = (float4)(0,0,255,255);
                break;
            }
            iter++;
        }
    }else{
        while(notfound){
    //        angle_step = 2*M_PI_F/((float)pow((float)(iter+1),2));
            totalAngles += 4;
            angle_step = (float)2*M_PI_F/(float)totalAngles;
            for(float i=0; i<2*M_PI_F; i+=angle_step){
                newx = ctrx + sin(i)*mag;
                newy = ctry + cos(i)*mag;
                if( newx >= 1 || newy >=1 || newx < 0 || newy < 0 )
                    continue;
                else{
                    gx = read_imageui(src, sampler, (float2)(newx,newy));
                    if(gx.x < 1){
                        newval.z = 30 + mag*255;
						newval.x = -mag*sqrt( (float)width*width + height*height);
                        notfound = false;
                        break;
                    }
                }
            }
            mag = mag + magstep;
            if(mag > maxMag){
                newval = (float4)(0,0,255,255);
                break;
            }
            iter++;
        }
    }

    write_imagef(dst, (int2)(col, row), newval);
}
