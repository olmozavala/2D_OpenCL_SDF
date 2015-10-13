
/**
 * For vectors X and Y it computes SUM of (x.i - y.i)^2     for i != dimension 
 * for the moment only works for dimension 1 and 2
 */
float getVectDistRestricted(int2 a, int2 b, int d){

    float dist;
	if(d == 1){
	   dist = (float)pow((float)a.y-b.y,2);
	}
	else {
	   dist = (float)pow((float)a.x-b.x,2);
	}
    return dist;
}

/**
 * It computes the L2 norm if 'squared' is false and
 * the L2^2 if 'squared' is true
 */
float getVectDist(int2 a, int2 b, bool squared ){
    float2 c = (float2)(a.x-b.x, a.y-b.y);
    float dist;

    if( squared)
        dist =  pow(c.x,2) + pow(c.y,2);
    else
        dist = sqrt( pow(c.x,2) + pow(c.y,2));

    return dist;
}

/**
 * Gets the 1D index from a 2D coordinates. (from (row,col) it gives a unique index
 */
int indxFromCoord(int width, int height, int row, int col, int dim){
    return width*row*dim + dim*col;
}

/**
 * It gets the two dimension coordinates from 1D index. From indx it gives the
 * corresponding row and column
 */
int2 getCoords(int indx, int width){

//	indx = indx - 1; //WARNING CHANGE THIS SO MAYBE THE OTHER CODE DOESN' WORK
    int2 coords = (int2)(0,0);
    coords.x = (int)floor((float) indx/(width) );
    coords.y = indx - coords.x*width;

    return coords;
}

/**
 * It receives  g[l-2] g[l-1] currIndx currcords
 */
bool removeFV(int ui, int vi, int wi, int2 curr_cell, int width, int d){
    int2 u = getCoords(ui,width);
    int2 v = getCoords(vi,width);
    int2 w = getCoords(wi,width);

    float a;
    float b;

    if(d==1){
        a = v.x - u.x;
        b = w.x - v.x;
    }else{
        a = v.y - u.y;
        b = w.y - v.y;
    }

    float c = a + b;

    float test_eqn =	  c*getVectDistRestricted(curr_cell, v, d)
                        - b*getVectDistRestricted(curr_cell, u, d)
                        - a*getVectDistRestricted(curr_cell, w, d)
                        - a*b*c;
	
	//Testing
//	float d1 = getVectDist(curr_cell,w,false);
//	float d2 = getVectDist(curr_cell,v,false);
//	float d3 = getVectDist(curr_cell,u,false);
//    if( (d2 >= d1) && (d2 >= d3) )

    if( test_eqn > 0)
        return true;
    else
        return false;
}

/**
 * This function returns the current 'pseudo row' that we are analyzing. 
 * if we are on dimension 1 then it returns a real row of the image
 * if we are on dimension 2 then it returns a column
 */
int2 getCurrCell(int curr_row, int curr_col, int cell, int d){
    if(d==1){
        return (int2)(cell,curr_col);
    }else{
        return (int2)(curr_row,cell);
    }
}

/* This function iterates of the second dimension and, in the case of 2D 
   computes the distance to the closes pixel to the oposite binary value */
__kernel void
SDF_voroStep2Buf(__global float* src, __global float* dst, int width, int height,  int d)
{
	// Maximum possible distance (from one corner to the other, used to normalize
	// the distances from 0 to 255)
	float maxDist = getVectDist( (int2)(0,0), (int2)(width,height), false);

    int curr_col = (int)get_global_id(0);
    int curr_row = (int)get_global_id(1);

    int rwidth = 0;//This will be the line width (dependent of current d)
	int g[4096];// TODO THIS ARRAY SHOULD BE THE SIZE OF MAX(width, height)

	//For d==1 internal loop is for rows and for d=2 internal is for columns
    if(d==1){
        rwidth = height;
    }
    else{
        rwidth = width;
    }

	int currIndx = 0; //Index used inside the loop, it iterates over rows or cols

    int l=0;
    float x = 0; //Value that we read from source
	float fij = 0; //Final result shortest distance to voronoi cell on that row

    int2 curr_cell = (int2)(0 , 0);//Only used to get the currIndx (from old functionality)

	//This is the internal loop 
    if(d==1){
        for(int cell=0; cell < rwidth; cell++){

            //This is the current cell we are analyzing for d=1 it iterates over a column
            curr_cell = (int2)(cell,curr_col);
            // Current cell has first row and then col we need to swap it
            currIndx = indxFromCoord(width, height, curr_cell.x, curr_cell.y, 1);

            x = src[currIndx];

            if( x > -1){
                if( l>1 ){
                   //This is the bottleneck of the program.
                   while( (l>1) && (removeFV( g[l-2] , g[l-1] , x , curr_cell ,
                                         width , d) )) {
                        l--;
                    }
                }
                g[l] = x;
                l++;
            }
        }
    }else{
        for(int cell=0; cell < rwidth; cell++){

            //This is the current cell we are analyzing for d=1 it iterates over a column
            curr_cell = (int2)(curr_row,cell);
            // Current cell has first row and then col we need to swap it
            currIndx = indxFromCoord(width, height, curr_cell.x, curr_cell.y, 1);

            x = src[currIndx];

            if( x > -1){
                if( l>1 ){
                   while( (l>1) && (removeFV( g[l-2] , g[l-1] , x , curr_cell ,
                                         width , d) )) {
                        l--;
                    }
                }
                g[l] = x;
                l++;
            }
        }
    }

	int totg = l-1;// Save total amount
    l = 0;

    int2 g1 = (int2)(0,0);
    int2 g2 = (int2)(0,0);
    bool doSquareRoot = false;// Avoids the squared root

    float dist1 = 0;
    float dist2 = 0;
	//totg = 0 means that there is one voronoi cell on this 'row'
    if( totg >= 0){
        for(int cell=0; cell < rwidth; cell++){
            curr_cell = getCurrCell(curr_row, curr_col, cell, d);
			currIndx = indxFromCoord(width, height, curr_cell.x, curr_cell.y, 1);

            if(totg > l){

                //Sets the current CFV and the next FV to see which one is the closest
                if( cell == 0){
                    g1 = getCoords(g[l], width);
                    g2 = getCoords(g[l+1], width);
                }

                if(  g[l+1] != currIndx ){
                    //(g[l+1] == currIndx) is to try to avoid the comparison
                    dist1 = getVectDist( curr_cell , g1, doSquareRoot );//To avoid computing it again
                    dist2 = getVectDist( curr_cell , g2, doSquareRoot );
                    while( (l < totg) && (dist1 > dist2)){
                        l = l+1;
                        if( totg > l){
                            g1 = getCoords(g[l] , width);
                            g2 = getCoords(g[l+1] , width);
                            dist1 = dist2;
                            dist2 = getVectDist( curr_cell , g2, doSquareRoot );
                        }
                    }
                }else{//This part of the if is to avoid computations, the normal algorithm is when if=true
                    l = l+1;
                    if( totg > l){
                        g1 = curr_cell;
                        g2 = getCoords(g[l+1] , width);
                        dist1 = 0;
                    }
                }
            }

			fij = 0;
			if( d == 1){
				fij = g[l];
			}
			else{ 
		// Write directly the distance in this case from 0 to max(width,height)
				fij = max(width,height)*getVectDist(curr_cell, getCoords(g[l], width), false)/maxDist;
                //fij = dist1;//This is not WORKING check it if you want to use it again
				//fij = g[l];
//                    fij = g[l]; //Used to show the closest index
//                    fij = totg; //Used to show the number of voronoi cell - 1 on this row 
//				fij = ((int2)getCoords(g[l], width)).x;
			}

			dst[currIndx]=fij;
		}
    }
    else{
		// If there is no FV in this row leave all the 'row' with -1's (Unknown FV)
		for(int cell=0; cell < rwidth; cell++){

            curr_cell = getCurrCell(curr_row, curr_col, cell, d);
			currIndx = indxFromCoord(width, height, curr_cell.x, curr_cell.y, 1);
			
			dst[currIndx]=-1;
		}
    }
}



__kernel void
SDF_voroStep2BufBackup(__global float* src, __global float* dst, int width, int height,  int d)
{
	// Maximum possible distance (from one corner to the other, used to normalize
	// the distances from 0 to 255)
	float maxDist = getVectDist( (int2)(0,0), (int2)(width,height), false);

    int curr_col = (int)get_global_id(0);
    int curr_row = (int)get_global_id(1);

    int rwidth = 0;//This will be the line width (dependent of current d)
	int g[4096];// TODO THIS ARRAY SHOULD BE THE SIZE OF MAX(width, height)

	//For d==1 internal loop is for rows and for d=2 internal is for columns
    if(d==1){
        rwidth = height;
    }
    else{
        rwidth = width;
    }

	int currIndx = 0; //Index used inside the loop, it iterates over rows or cols

    int l=0;
    float x = 0; //Value that we read from source
	float fij = 0; //Final result shortest distance to voronoi cell on that row

    int2 curr_cell = (int2)(0 , 0);//Only used to get the currIndx (from old functionality)

	//This is the internal loop 
    for(int cell=0; cell < rwidth; cell++){

        //This is the current cell we are analyzing for d=1 it iterates over a column
        curr_cell = getCurrCell(curr_row, curr_col, cell, d);

		// Current cell has first row and then col we need to swap it
		currIndx = indxFromCoord(width, height, curr_cell.x, curr_cell.y, 1);

		x = src[currIndx];

        if( x > -1){
            if( l>1 ){
               while( (l>1) && (removeFV( g[l-2] , g[l-1] , x , curr_cell ,
                                     width , d) )) {
                    l--;
                }
			}
			g[l] = x;
			l++;
        }
    }

	int totg = l-1;// Save total amount
    l = 0;

    int2 g1 = (int2)(0,0);
    int2 g2 = (int2)(0,0);
    bool doSquareRoot = false;// Avoids the squared root

	//totg = 0 means that there is one voronoi cell on this 'row'
    if( totg >= 0){
        for(int cell=0; cell < rwidth; cell++){
            curr_cell = getCurrCell(curr_row, curr_col, cell, d);
			currIndx = indxFromCoord(width, height, curr_cell.x, curr_cell.y, 1);

            if(totg > l){

                //Sets the current CFV and the next FV to see which one is closest
                g1 = getCoords(g[l], width);
                g2 = getCoords(g[l+1], width);

                while( (l < totg) &&
                            ( getVectDist( curr_cell , g1, doSquareRoot ) >
                                        getVectDist( curr_cell , g2, doSquareRoot ))){
                    l = l+1;
                    if( totg > l){
                        g1 = getCoords(g[l] , width);
                        g2 = getCoords(g[l+1] , width);
                    }
                }
            }

			fij = 0;
			if( d == 1){
				fij = g[l];
			}
			else{ 
		// Write directly the distance in this case from 0 to max(width,height)
//				fij = max(width,height)*getVectDist(curr_cell, getCoords(g[l], width), false)/maxDist;
				fij = getVectDist(curr_cell,getCoords(g[l], width), false);
//                    fij = g[l]; //Used to show the closest index
//                    fij = totg; //Used to show the number of voronoi cell - 1 on this row 
//				fij = ((int2)getCoords(g[l], width)).x;
			}

			dst[currIndx]=fij;
		}
    }
    else{
		// If there is no FV in this row leave all the 'row' with -1's (Unknown FV)
		for(int cell=0; cell < rwidth; cell++){

            curr_cell = getCurrCell(curr_row, curr_col, cell, d);
			currIndx = indxFromCoord(width, height, curr_cell.x, curr_cell.y, 1);
			
			dst[currIndx]=-1;
		}
    }
}

__kernel void
mergePhisBuf( __global float* buf_phi_half,__global float* buf_phi_sec_half,
			__global float*buf_phi,int width, int height){
				
	int col = get_global_id(0);
	int row = get_global_id(1);

	int currIndx = indxFromCoord(width, height, row, col, 1);

	float fhalf = buf_phi_half[currIndx];
	float sechalf = buf_phi_sec_half[currIndx];

	if( sechalf > -1){
		//The plus 1 is in order to have a band of 0
		buf_phi[currIndx] = fhalf - sechalf + 1;
	}else{
		buf_phi[currIndx] = fhalf;
	}
}

/**
 * Fills a new image with the values of a '1D index' inside all the
 * pixels that has value > 0 if mode == 1 or for all the pixels
 * that has value < 0 if mode == 0
 */
__kernel void
SDF_voroStep1Buf(__global uchar* src, __global float* dst, int width, int height, int mode)
{
    int col = (int)get_global_id(0);
    int row = (int)get_global_id(1);

	int  currIndx = indxFromCoord(width, height, row, col, 1);
    uchar x = src[currIndx];

	if(currIndx <= width*height){//Checking boundaries
		if( mode == 1){ // Distance to values >  0
			if(x > 0){
				dst[currIndx] = currIndx;
			}
			else{
				dst[currIndx] = -1;
			}
		}

		if(mode == 0){ // Distance to values == 0
			if(x == 0){
				dst[currIndx] = currIndx;
			}
			else{
				dst[currIndx] = -1;
			}
		}
	}
}
