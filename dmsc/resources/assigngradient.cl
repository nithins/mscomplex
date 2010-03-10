////////////////////////////////////////////////////////////////////////////////

// Gradient assignment computation kernel
//

#define cell_fn_t    float
#define cell_coord_t short
#define cell_flag_t  uchar

__kernel void assign_gradient
( __global cell_fn_t    *cell_fns,      
  __global cell_coord_t *cell_pairs,
  __global cell_flag_t  *cell_flags,     
   const cell_coord_t x_min,
   const cell_coord_t x_max,
   const cell_coord_t y_min,   
   const cell_coord_t y_max)
   
{  
   cell_coord_t x = x_min + get_global_id(0);
   cell_coord_t y = y_min + get_global_id(1);
   
   if(x > x_max || y > y_max)
     return;
   
   uint cell_op_idx = (y-y_min)*(x_max - x_min+1) + (x-x_min);   
   
   cell_pairs[cell_op_idx*2+0] = 1;
   cell_pairs[cell_op_idx*2+1] = 1;
   
   cell_flags[cell_op_idx] = 2;
}                              


