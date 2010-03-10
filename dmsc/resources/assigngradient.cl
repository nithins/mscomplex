// gradient assignment compute kernel
//

__kernel void assign_gradient
( __global float* pt_fns,    
  __global short* cell_pairs,
  __global uchar* cell_flags) 
{  
//   size_t x_max = get_global_size (0);
   //size_t y_max = get_global_size (1);	
   
//   uint x = get_global_id(0);
//   uint y = get_global_id(1);
   
   //cell_flags[y*x_max+x] = 2; // mark everyone as critical
}                                                                  
