////////////////////////////////////////////////////////////////////////////////

// Gradient assignment computation kernel
//


__kernel void assign_gradient
( __global float* input,      
  __global short* output,     
   const unsigned int count)   
{                              
   int i = get_global_id(0);   
       output[i] = 1;          
}                              


