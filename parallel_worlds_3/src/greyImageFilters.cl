        /* start the kernel with one work-item per pixel
        ** first work-dimension (0) is image width (x)
        */
        __kernel void grey(__global unsigned char *inImg,
                            __global unsigned char *outImg,
                            __private unsigned int w,__private unsigned int h) {
            __private unsigned int x;__private unsigned int y;
            x = get_global_id(0);
            y = get_global_id(1);
            if(y<h) {
                if(x<w) {
                    // greyscale conversion (c.f. http://en.wikipedia.org/wiki/Grayscale)
                    // Y = 0.2126R + 0.7152G + 0.0722B
                    outImg[x+w*y] = 0.0722 * inImg[3*(x+w*y)] /* blue */
                    + 0.7152 * inImg[3*(x+w*y)+1]             /* green */
                    + 0.2126 * inImg[3*(x+w*y)+2];            /* red */
                }
            }
        }
        #define ARR(A,x,y,maxX) (A[(x)+(y)*(maxX)])
        /// sobel filter (cf. http://en.wikipedia.org/wiki/Sobel_operator):
        // detect edges by computing the convolution
        // with matrix {{-1,0,1},{-2,0,2},{-1,0,1}} in x- and y- direction;
        // the result is computed as c*sqrt(G_x^2 + G_y^2) (where G_x/G_y
        // is the convolution with the above matrix);
        // this computation is only done for interior pixels -- the edges
        // of the image are blacked out;
        // @param inImg pointer to the input grey image in device memory
        // @param outImg pointer to the output grey image in device memory
        // @param w width of image
        // @param h height of image
        // @param c coefficient by which to multiply the actual convolution
        // @param img image portion for computation -- to be shared between
        //          work-items of a work-group (each work-item writes exactly
        //          1 pixel of img)
        // Note: img has to be passed via Kernel::setArg(), because its size
        // depends on the size of the work-group (otherwise it could have been
        // defined inside the kernel)
        __kernel void sobel(__global unsigned char *inImg,
                            __global unsigned char *outImg,
                            unsigned int w,unsigned int h,
                            float c,
                            __local unsigned char *img){
            // coordinates of input pixel in cache array img
            unsigned int xCache;unsigned  int yCache;
            // coordinates of pixel in input/output image
            unsigned int x;unsigned  int y;
            // number of output pixels per work-group in x/y direction
            // will evaluate to 8, since the kernel will be started on a
            // 10 * 10 work-group
            unsigned int numOutX; unsigned int numOutY;
            numOutX = get_local_size(0) - 2; numOutY = get_local_size(1) - 2;
            x = get_group_id(0) * numOutX + get_local_id(0);
            y = get_group_id(1) * numOutY + get_local_id(1);
            xCache = get_local_id(0); yCache = get_local_id(1);
            if(x<w && y<h){
                // read pixels from original image into cache
                ARR(img,xCache,yCache,get_local_size(0)) = ARR(inImg,x,y,w);
                // border pixels are all black
                if(0==x||0==y||w-1==x||h-1==y){
                    ARR(outImg,x,y,w) = 0;
                }
             }
             // wait for all work-items to finish copying
             barrier(CLK_LOCAL_MEM_FENCE);
             if(x<w-1 && y<h-1){
                // compute result value and write it back to device memory
                // (but only for interior pixels, i.e. 1<=id<=max-1)
                if(xCache > 0 && xCache < get_local_size(0) - 1){
                    if(yCache > 0 && yCache < get_local_size(1) - 1){
                        __private float G_x =
                                    -ARR(img,xCache-1,yCache-1,get_local_size(0))
                                    -2*ARR(img,xCache-1,yCache,get_local_size(0))
                                    -ARR(img,xCache-1,yCache+1,get_local_size(0))
                                    +ARR(img,xCache+1,yCache-1,get_local_size(0))
                                    +2*ARR(img,xCache+1,yCache,get_local_size(0))
                                    +ARR(img,xCache+1,yCache+1,get_local_size(0));
                        __private float G_y =
                                    -ARR(img,xCache-1,yCache-1,get_local_size(0))
                                    -2*ARR(img,xCache,yCache-1,get_local_size(0))
                                    -ARR(img,xCache+1,yCache-1,get_local_size(0))
                                    +ARR(img,xCache-1,yCache+1,get_local_size(0))
                                    +2*ARR(img,xCache,yCache+1,get_local_size(0))
                                    +ARR(img,xCache+1,yCache+1,get_local_size(0));
                        // sqrt is a predefined OpenCL function!
                        ARR(outImg,x,y,w) = (unsigned char) (c * sqrt(G_x*G_x + G_y*G_y));
                    }
                }
            }
        }
