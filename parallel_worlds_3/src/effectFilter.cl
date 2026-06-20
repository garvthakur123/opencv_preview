#define ARR(A,x,y,maxX) (A[(x)+(y)*(maxX)])
#define ARRC(A,x,y,maxX,channel) (A[((x)+(y)*(maxX))*3+channel])
/// effect filter based on a sobel edge detection;
// @param inOutImg pointer to the color image in device memory -- this
// image is used for input AND output
// @param edgeImg pointer to the input color edge image in device memory
// @param w width of image
// @param h height of image
// @param threshold threshold for drawing rays
void __kernel effectFilter(__global unsigned char *inOutImg,
                           __global char *edgeImg,
                            unsigned int w,unsigned int h,
                            float threshold){
  unsigned int x = get_global_id(0);
  unsigned int y = get_global_id(1);

  if(x<w && y<h){
    float G = ((char)ARR(edgeImg,x,y,w));
    float absG = fabs(G);
    if(absG > threshold){
        ARRC(inOutImg,x,y,w,0) = (int)(ARRC(inOutImg,x,y,w,0)*.5);
        ARRC(inOutImg,x,y,w,1) = (int)(ARRC(inOutImg,x,y,w,1)*.5);
        ARRC(inOutImg,x,y,w,2) = (int)(ARRC(inOutImg,x,y,w,2)*.5);
    }
  }
}
