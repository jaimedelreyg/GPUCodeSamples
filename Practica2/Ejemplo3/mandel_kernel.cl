// Mandel_kernel.cl
// Kernel source file for calculating mandelbrot fractal
// Author: Jaime del Rey

_kernel
void mandel_fractal(__global float ** tex,
		    __global float scale,
		    __global float width,
		    __global float height,
		    __global float cx,
		    __global float cy)

{

int i, j, iter;
double x, y, zx, zy, zx2, zy2;
double t0;

i = get_global_id(0);
j = get_global_id(1);


y = (i - height/2) * scale + cy;
x = (j - width/2) * scale + cx;

zx = zy = zx2 = zy2 = 0;


for (iter=0; iter < max_iter; iter++) {
       zy=2*zx*zy + y;
       zx=zx2-zy2 + x;
       zx2=zx*zx;
       zy2=zy*zy;
       if (zx2+zy2>max_iter)
       	break;
}

*(unsigned short *)&(tex[i][j]) = iter;


}

