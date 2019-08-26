//OPENCL KERNEL FOR PI CALCULATION

__kernel
void pi(__global double* parcial_sums, uint n)
{

int id = get_global_id(0);
int i = 0;
double x, area;

area= 0.0;

x = (id+0.5)/n;
area += 4.0/(1.0 + x*x);

parcial_sums[id] = area;

}
