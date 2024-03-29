// Mandel_kernel.cl
// Kernel source file for calculating mandelbrot fractal
// Author: Jaime del Rey

typedef struct {unsigned char r, g, b;} rgb_t;

void hsv_to_rgb(int hue, int min, int max, __global rgb_t *p, int color_rotate, int saturation,
int invert)
{
	int val = 255;

	if (min == max) max = min + 1;
	if (invert) hue = max - (hue - min);
	if (!saturation) {
		p->r = p->g = p->b = 255 * (max - hue) / (max - min);
		return;
	}
	double h = fmod(color_rotate + 1e-4 + 4.0 * (hue - min) / (max - min), 6);
	double c = val * saturation;
	double X = c * (1 - fabs(fmod(h, 2) - 1));
 
	p->r = p->g = p->b = 0;
 
	switch((int)h) {
	case 0: p->r = c; p->g = X; return;
	case 1: p->r = X; p->g = c; return;
	case 2: p->g = c; p->b = X; return;
	case 3: p->g = X; p->b = c; return;
	case 4: p->r = X; p->b = c; return;
	default:p->r = c; p->b = X;
	}
}


__kernel
void mandel_fractal(__global rgb_t *tex,
		    double scale,
		    int tex_width,
		    int height,
		    int width,
		    double cx,
		    double cy,
		    int max_iter, int saturation,
		    int invert, int color_rotate)

{

int i, j;
short iter;
double x, y, zx, zy, zx2, zy2;

i = (int)get_global_id(0);
j = (int)get_global_id(1);

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

hsv_to_rgb(iter, 0, max_iter, &tex[i*width+j], color_rotate, saturation, invert);

}
