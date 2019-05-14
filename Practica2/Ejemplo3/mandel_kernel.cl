// Mandel_kernel.cl
// Kernel source file for calculating mandelbrot fractal
// Author: Jaime del Rey


void hsv_to_rgb(int hue, int min, int max, __global rgb_t *p)
{
	int color_rotate = 0;
	int saturation = 1;
	int invert = 0;

	if (min == max) max = min + 1;
	if (invert) hue = max - (hue - min);
	if (!saturation) {
		p->r = p->g = p->b = 255 * (max - hue) / (max - min);
		return;
	}
	double h = fmod(color_rotate + 1e-4 + 4.0 * (hue - min) / (max - min), 6);
#	define VAL 255
	double c = VAL * saturation;
	double X = c * (1 - fabs(fmod(h, 2) - 1));
 
	p[0] = p[1] = p[2] = 0;
 
	switch((int)h) {
	case 0: p[0] = c; p[1] = X; return;
	case 1: p[0] = X; p[1] = c; return;
	case 2: p[0] = c; p[1] = X; return;
	case 3: p[0] = X; p[1] = c; return;
	case 4: p[0] = X; p[1] = c; return;
	default:p[0] = c; p[1] = X;
	}
}

__kernel
void mandel_fractal(__global char *tex,
		    const uint scale,
		    const uint width,
		    const uint height,
		    const double cx,
		    const double cy,
		    const uint max_iter)

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


hsv_to_rgb(iter, 0, max_iter, &tex[(i * width + j) * 3]);

}
