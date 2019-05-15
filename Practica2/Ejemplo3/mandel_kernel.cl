// Mandel_kernel.cl
// Kernel source file for calculating mandelbrot fractal
// Author: Jaime del Rey


typedef struct {unsigned char r, g, b;} rgb_t;

__kernel
void mandel_fractal(__global uint *tex,
		    const double scale,
		    const uint width,
		    const uint height,
		    const double cx,
		    const double cy,
		    const int max_iter)

{

double i, j, iter, x, y, zx, zy, zx2, zy2, min, val;

min = 0.0;
val = 255.0;

i = get_global_id(0);
j = get_global_id(1);


y = (i - height/2) * scale + cy;
x = (j - width/2) * scale + cx;

zx = zy = zx2 = zy2 = 0.0;

for (iter=0; iter < max_iter; iter = iter+1.0) {
       zy=2*zx*zy + y;
       zx=zx2-zy2 + x;
       zx2=zx*zx;
       zy2=zy*zy;
       if (zx2+zy2>max_iter)
       	break;

}

tex[(int)(i*width+j)] = iter;

}
