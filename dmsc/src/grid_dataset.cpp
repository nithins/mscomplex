#include <grid_dataset.h>

#include <discreteMorseAlgorithm.h>
#include <vector>

#include <timer.h>

#include <QFile>
#include <prefix_scan.h>

typedef GridDataset::cellid_t cellid_t;

cl_device_id s_device_id;             // compute device id
cl_context s_context;                 // compute context
cl_program s_grad_pgm;                // compute program
cl_program s_coll_cps_pgm;            // compute program
PrefixScan s_pre_scan;                // prefix scanning program



const int max_threads_1D   = 128;
const int max_threads_2D_x = 16;
const int max_threads_2D_y = 16;

#define _CHECKCL_ERR_CODE(_ERROR,_MESSAGE)\
if(_ERROR != CL_SUCCESS) throw std::runtime_error(_MESSAGE);

void log_cl_compilation_error(cl_program &prog,const char * ptx_file_name= NULL)
{
  size_t len;
  char buffer[2048];

  clGetProgramBuildInfo(prog, s_device_id, CL_PROGRAM_BUILD_LOG,
                        sizeof(buffer), buffer, &len);

  if(ptx_file_name != NULL)
  {
    // Log the binary generated

    size_t bin_size;
    clGetProgramInfo(prog,CL_PROGRAM_BINARY_SIZES,
                     sizeof(size_t),&bin_size,NULL);

    char * ptx_buffer = new char[bin_size];

    clGetProgramInfo(prog,CL_PROGRAM_BINARIES,
                     sizeof(ptx_buffer),&ptx_buffer,NULL);

    _LOG_TO_FILE(std::string(ptx_buffer),ptx_file_name);

    delete []ptx_buffer;
  }

  throw std::runtime_error(buffer);
}

void GridDataset::init_opencl()
{

  int error_code;                            // error code returned from api calls

  // Connect to a compute device
  //
  error_code = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU , 1, &s_device_id, NULL);
  if (error_code != CL_SUCCESS)
    throw std::runtime_error("Error: Failed to create a device group!\n");

  // Create a compute context
  //
  s_context = clCreateContext(0, 1, &s_device_id, NULL, NULL, &error_code);

  _CHECKCL_ERR_CODE(error_code,"Failed to create a compute context");

  // Create the gradient assignment compute program from the source buffer
  //

  QFile ocl_assign_grad_src ( ":/oclsources/assigngradient.cl" );
  ocl_assign_grad_src.open(QIODevice::ReadOnly);

  QByteArray assign_grad_src = ocl_assign_grad_src.readAll().constData();

  const char * assign_grad_chptr = assign_grad_src.constData();

  s_grad_pgm = clCreateProgramWithSource
               (s_context, 1, (const char **) & assign_grad_chptr,
                NULL, &error_code);

  _CHECKCL_ERR_CODE(error_code,"Failed to create assign gradient program");

  // Build the program executable
  //
  error_code = clBuildProgram(s_grad_pgm, 0, NULL, "-cl-opt-disable", NULL, NULL);

  if (error_code != CL_SUCCESS)
    log_cl_compilation_error(s_grad_pgm,"assign_grad.ptx");


  // Create the critical point collation compute program from the source buffer
  //

  QFile ocl_create_critpts_src ( ":/oclsources/collate_critpts.cl" );
  ocl_create_critpts_src.open(QIODevice::ReadOnly);

  QByteArray collate_critpts_src = ocl_create_critpts_src.readAll().constData();

  const char * collate_critpts_chptr = collate_critpts_src.constData();

  s_coll_cps_pgm
      = clCreateProgramWithSource(s_context, 1, & collate_critpts_chptr,
                                  NULL, &error_code);

  _CHECKCL_ERR_CODE(error_code,"Failed to create collate critpts program");

  // Build the program executable
  //
  error_code = clBuildProgram(s_coll_cps_pgm, 0, NULL, "-cl-opt-disable", NULL, NULL);
  if (error_code != CL_SUCCESS)
    log_cl_compilation_error(s_coll_cps_pgm,"assign_grad.ptx");


  s_pre_scan.init(s_context,s_device_id);
}

void GridDataset::stop_opencl()
{
  // Shutdown and cleanup
  //
  clReleaseProgram(s_coll_cps_pgm);
  clReleaseProgram(s_grad_pgm);
  clReleaseContext(s_context);
  s_pre_scan.cleanup();

}

void  GridDataset::create_pair_flag_imgs_ocl()
{

  rect_size_t sz = m_ext_rect.size();

  size_t cell_img_rgn[3] = {sz[0]+1,sz[1]+1,1};

  int error_code;                       // error code returned from api calls

  cl_image_format cell_pr_imgfmt,cell_fg_imgfmt;

  cell_pr_imgfmt.image_channel_data_type = CL_SIGNED_INT16;
  cell_pr_imgfmt.image_channel_order     = CL_RG;

  cell_fg_imgfmt.image_channel_data_type = CL_UNSIGNED_INT8;
  cell_fg_imgfmt.image_channel_order     = CL_R;

  m_cell_pair_img = clCreateImage2D
                    (s_context,CL_MEM_WRITE_ONLY,
                     &cell_pr_imgfmt,cell_img_rgn[0],cell_img_rgn[1],0,
                     NULL,&error_code);

  _CHECKCL_ERR_CODE(error_code,"Failed to create cell pair image");

  m_cell_flag_img = clCreateImage2D
                    (s_context,CL_MEM_WRITE_ONLY,
                     &cell_fg_imgfmt,cell_img_rgn[0],cell_img_rgn[1],0,
                     NULL,&error_code);

  _CHECKCL_ERR_CODE(error_code,"Failed to create cell flag texture");

}

void  GridDataset::read_pair_flag_imgs_ocl(cl_command_queue &commands)
{

  rect_size_t sz = m_ext_rect.size();

  size_t cell_img_ogn[3] = {0,0,0};
  size_t cell_img_rgn[3] = {sz[0]+1,sz[1]+1,1};

  int error_code;

  // Read back the results from the device to verify the output
  //

  error_code = clEnqueueReadImage( commands, m_cell_pair_img, CL_TRUE,
                                   cell_img_ogn,cell_img_rgn,0,0,
                                   m_cell_pairs.data(),0,NULL,NULL);

  _CHECKCL_ERR_CODE(error_code,"Failed to read back cell pair image");

  error_code = clEnqueueReadImage( commands, m_cell_flag_img, CL_TRUE,
                                   cell_img_ogn,cell_img_rgn,0,0,
                                   m_cell_flags.data(),0,NULL,NULL);

  _CHECKCL_ERR_CODE(error_code,"Failed to read back cell flag image");


}

void  GridDataset::clear_pair_flag_imgs_ocl()
{
  clReleaseMemObject(m_cell_pair_img);
  clReleaseMemObject(m_cell_flag_img);
}

#define _GET_GLOBAL(s,l)\
(((s)/(2*(l)))*(2*(l)) + ((((s)%(2*(l))) == 0)?(0):(2*l)))


    void GridDataset::assignGradients_ocl()
{

  create_pair_flag_imgs_ocl();

  Timer timer;

  timer.start();

  int error_code;                       // error code returned from api calls

  rect_size_t sz = m_ext_rect.size();

  size_t vert_img_rgn[3]  = {(sz[0]>>1)+1,(sz[1]>>1)+1,1};

  size_t local[] = {max_threads_2D_x,max_threads_2D_y};

  size_t global[] =
  {
    _GET_GLOBAL(sz[0]+1,local[0]) ,
    _GET_GLOBAL(sz[1]+1,local[1]) ,
  }; // should be div by 2*local

  // Create a command commands
  //
  cl_command_queue commands = clCreateCommandQueue
                              (s_context, s_device_id, 0, &error_code);

  _CHECKCL_ERR_CODE(error_code,"Failed to create commands queue")

  // Create the compute kernel in the program we wish to run
  //
  cl_kernel kernel = clCreateKernel(s_grad_pgm, "assign_gradient", &error_code);

  _CHECKCL_ERR_CODE(error_code,"Failed to create assign gradient kernel")

  // Create the input and output arrays in device memory for our calculation
  //

  cl_image_format vert_fn_imgfmt;

  vert_fn_imgfmt.image_channel_data_type = CL_FLOAT;
  vert_fn_imgfmt.image_channel_order     = CL_R;

  cl_mem vfn_img_cl = clCreateImage2D
                      (s_context,CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR,
                       &vert_fn_imgfmt,vert_img_rgn[0],vert_img_rgn[1],0,
                       m_vertex_fns.data(),&error_code);

  _CHECKCL_ERR_CODE(error_code,"Failed to create cell fn texture")

  _LOG("Done tranfer Data    t = "<<timer.getElapsedTimeInMilliSec()<<" ms");

  cell_coord_t x_min = m_ext_rect.left()  ,x_max = m_ext_rect.right();
  cell_coord_t y_min = m_ext_rect.bottom(),y_max = m_ext_rect.top();

  // Set the arguments to our compute kernel
  //
  error_code = 0;
  error_code  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &vfn_img_cl);
  error_code |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &m_cell_pair_img);
  error_code |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &m_cell_flag_img);
  error_code |= clSetKernelArg(kernel, 3, sizeof(cell_coord_t), &x_min);
  error_code |= clSetKernelArg(kernel, 4, sizeof(cell_coord_t), &x_max);
  error_code |= clSetKernelArg(kernel, 5, sizeof(cell_coord_t), &y_min);
  error_code |= clSetKernelArg(kernel, 6, sizeof(cell_coord_t), &y_max);

  _CHECKCL_ERR_CODE(error_code,"Failed to set assign grad kern args")

  error_code = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
                                      global, local, 0, NULL, NULL);

  _CHECKCL_ERR_CODE(error_code,"Failed to enque assign grad kernel")

  // Wait for the command commands to get serviced before launching next kernel
  //
  clFinish(commands);

  _LOG("Done gradient part 1 t = "<<timer.getElapsedTimeInMilliSec()<<" ms");

  // Release what we dont need anymore
  //
  clReleaseMemObject(vfn_img_cl);
  clReleaseKernel(kernel);

  kernel = clCreateKernel(s_grad_pgm, "complete_pairings", &error_code);

  _CHECKCL_ERR_CODE(error_code,"Failed to create complete_pairings kernel")

  // Set the arguments to our compute kernel
  //
  error_code = 0;
  error_code  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &m_cell_pair_img);
  error_code |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &m_cell_pair_img);
  error_code |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &m_cell_flag_img);
  error_code |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &m_cell_flag_img);
  error_code |= clSetKernelArg(kernel, 4, sizeof(cell_coord_t), &x_min);
  error_code |= clSetKernelArg(kernel, 5, sizeof(cell_coord_t), &x_max);
  error_code |= clSetKernelArg(kernel, 6, sizeof(cell_coord_t), &y_min);
  error_code |= clSetKernelArg(kernel, 7, sizeof(cell_coord_t), &y_max);

  _CHECKCL_ERR_CODE(error_code,"Failed to set complete_pairings arguments")

  error_code = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
                                      global, local, 0, NULL, NULL);

  _CHECKCL_ERR_CODE(error_code,"Failed to enqueue complete_pairings kernel")

  // Wait for the command commands to get serviced before copying results
  //
  clFinish(commands);

  _LOG("Done gradient part 2 t = "<<timer.getElapsedTimeInMilliSec()<<" ms");

  read_pair_flag_imgs_ocl(commands);

  _LOG("Done reading results t = "<<timer.getElapsedTimeInMilliSec()<<" ms");

  // Release stuff we dont need anymore
  //

  clReleaseKernel(kernel);

  _LOG("Done grad assignment t = "<<timer.getElapsedTimeInMilliSec()<<" ms");

  collateCritcalPoints_ocl(commands);

  _LOG("Done cp collation    t = "<<timer.getElapsedTimeInMilliSec()<<" ms");

  assignCellOwnerExtrema_ocl(commands);

  clear_pair_flag_imgs_ocl();

  clReleaseCommandQueue(commands);

}

void GridDataset::collateCritcalPoints_ocl(cl_command_queue &commands)
{
  int error_code;

  rect_size_t sz = m_ext_rect.size();

  int grid_size = (sz[0]+1)*(sz[1]+1);

  int critpt_idx_buf_sz = grid_size*sizeof(uint);

  cl_kernel kernel;

  // Get the maximum work group size for executing the kernel on the device
  // TODO: get a good value for each kernel
  size_t local[] = {max_threads_2D_x,max_threads_2D_y};

  // Execute the kernel over the entire range of our 1d input data set
  // using the maximum number of work group items for this device
  //
  size_t global[] =
  {
    _GET_GLOBAL(sz[0]+1,local[0]) ,
    _GET_GLOBAL(sz[1]+1,local[1]) ,
  }; // should be div by 2*local

  cell_coord_t x_min = m_ext_rect.left()  ,x_max = m_ext_rect.right();
  cell_coord_t y_min = m_ext_rect.bottom(),y_max = m_ext_rect.top();

  cl_mem critpt_idx_buf =
      clCreateBuffer(s_context,CL_MEM_READ_WRITE,critpt_idx_buf_sz,NULL,&error_code);

  _CHECKCL_ERR_CODE(error_code,"couldnt create prefixsum_buf");

  kernel = clCreateKernel(s_coll_cps_pgm, "collate_cps_initcount", &error_code);

  _CHECKCL_ERR_CODE(error_code,"Failed to create compute kernel");

  // Set the arguments to our compute kernel
  //
  uint a=0;

  error_code = 0;
  error_code  = clSetKernelArg(kernel, a++, sizeof(cl_mem), &m_cell_flag_img);
  error_code |= clSetKernelArg(kernel, a++, sizeof(cl_mem), &critpt_idx_buf);
  error_code |= clSetKernelArg(kernel, a++, sizeof(cell_coord_t), &x_min);
  error_code |= clSetKernelArg(kernel, a++, sizeof(cell_coord_t), &x_max);
  error_code |= clSetKernelArg(kernel, a++, sizeof(cell_coord_t), &y_min);
  error_code |= clSetKernelArg(kernel, a++, sizeof(cell_coord_t), &y_max);

  _CHECKCL_ERR_CODE(error_code,"Failed to set kernel arguments");

  error_code = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
                                      global, local, 0, NULL, NULL);

  _CHECKCL_ERR_CODE(error_code,"Failed to execute kernel");

  clFinish(commands);

  clReleaseKernel(kernel);

  s_pre_scan.CreatePartialSumBuffers(grid_size,s_context);
  s_pre_scan.PreScanBuffer(critpt_idx_buf,critpt_idx_buf,grid_size,commands);
  s_pre_scan.ReleasePartialSums();


  _CHECKCL_ERR_CODE(error_code,"Failed to execute kernel");

  uint crit_pt_ct = 0 ;

  clEnqueueReadBuffer(commands,critpt_idx_buf,CL_TRUE,critpt_idx_buf_sz-sizeof(uint),
                      sizeof(uint),&crit_pt_ct,0,NULL,NULL);

  crit_pt_ct++;

  uint crit_pt_id_buf_sz = crit_pt_ct*sizeof(cell_coord_t)*2;

  cl_mem critpt_id_buf = clCreateBuffer(s_context,CL_MEM_READ_WRITE,
                                        crit_pt_id_buf_sz,NULL,&error_code);

  _CHECKCL_ERR_CODE(error_code,"Failed to create crit_pt_id_buf");

  kernel = clCreateKernel(s_coll_cps_pgm, "collate_cps_writeids", &error_code);

  _CHECKCL_ERR_CODE(error_code,"Failed to create compute kernel");

  // Set the arguments to our compute kernel
  //
  error_code = 0;
  error_code  = clSetKernelArg(kernel, a++, sizeof(cl_mem), &m_cell_flag_img);
  error_code |= clSetKernelArg(kernel, a++, sizeof(cl_mem), &critpt_idx_buf);
  error_code |= clSetKernelArg(kernel, a++, sizeof(cl_mem), &critpt_id_buf);
  error_code |= clSetKernelArg(kernel, a++, sizeof(cell_coord_t), &x_min);
  error_code |= clSetKernelArg(kernel, a++, sizeof(cell_coord_t), &x_max);
  error_code |= clSetKernelArg(kernel, a++, sizeof(cell_coord_t), &y_min);
  error_code |= clSetKernelArg(kernel, a++, sizeof(cell_coord_t), &y_max);

  _CHECKCL_ERR_CODE(error_code,"Failed to set kernel arguments");

  error_code = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
                                      global, local, 0, NULL, NULL);

  _CHECKCL_ERR_CODE(error_code,"Failed to execute kernel");

  clFinish(commands);

  clReleaseKernel(kernel);

  m_critical_cells.resize(crit_pt_ct);

  clEnqueueReadBuffer(commands,critpt_id_buf,CL_TRUE,0,
                      crit_pt_id_buf_sz,m_critical_cells.data(),0,NULL,NULL);

  clReleaseMemObject(critpt_idx_buf);
  clReleaseMemObject(critpt_id_buf);
}

void GridDataset::assignCellOwnerExtrema_ocl(cl_command_queue &commands)
{

  cl_kernel kernel;

  rect_size_t sz = m_ext_rect.size();

  size_t cell_img_rgn[3] = {sz[0]+1,sz[1]+1,1};
  size_t cell_img_ogn[3] = {0,0,0};

  int error_code;                       // error code returned from api calls

  // Get the maximum work group size for executing the kernel on the device
  // TODO: get a good value for each kernel
  size_t local[] = {max_threads_2D_x,max_threads_2D_y};

  // Execute the kernel over the entire range of our 1d input data set
  // using the maximum number of work group items for this device
  //
  size_t global[] =
  {
    _GET_GLOBAL(sz[0]+1,local[0]) ,
    _GET_GLOBAL(sz[1]+1,local[1]) ,
  }; // should be div by 2*local

  cell_coord_t x_min = m_ext_rect.left()  ,x_max = m_ext_rect.right();
  cell_coord_t y_min = m_ext_rect.bottom(),y_max = m_ext_rect.top();

  cl_image_format cell_own_imgfmt;

  cell_own_imgfmt.image_channel_data_type = CL_SIGNED_INT16;
  cell_own_imgfmt.image_channel_order     = CL_RG;

  cl_mem cell_own_img = clCreateImage2D
                        (s_context,CL_MEM_READ_WRITE,
                         &cell_own_imgfmt,cell_img_rgn[0],cell_img_rgn[1],0,
                         NULL,&error_code);

  _CHECKCL_ERR_CODE(error_code,"Failed to create owner image");

  kernel = clCreateKernel(s_grad_pgm, "dobfs_markowner_extrema_init", &error_code);

  _CHECKCL_ERR_CODE(error_code,"Failed to create dobfs_init kernel");

  // Set the arguments to our compute kernel
  //
  uint a = 0 ;

  error_code  = 0;
  error_code  = clSetKernelArg(kernel, a++, sizeof(cl_mem), &m_cell_flag_img);
  error_code |= clSetKernelArg(kernel, a++, sizeof(cl_mem), &cell_own_img);
  error_code |= clSetKernelArg(kernel, a++, sizeof(cell_coord_t), &x_min);
  error_code |= clSetKernelArg(kernel, a++, sizeof(cell_coord_t), &x_max);
  error_code |= clSetKernelArg(kernel, a++, sizeof(cell_coord_t), &y_min);
  error_code |= clSetKernelArg(kernel, a++, sizeof(cell_coord_t), &y_max);

  _CHECKCL_ERR_CODE(error_code,"Failed to set args for dobfs_init kernel");

  error_code = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
                                      global, local, 0, NULL, NULL);

  _CHECKCL_ERR_CODE(error_code,"Failed to execute dobfs_init kernel");

  clFinish(commands);

  uint is_changed = 0;

  kernel = clCreateKernel(s_grad_pgm, "dobfs_markowner_extrema", &error_code);

  _CHECKCL_ERR_CODE(error_code,"Failed to create dobfs_markowner_extrema kernel");

  uint iteration_ct = 0;

  cl_mem is_changed_buf =
      clCreateBuffer(s_context,CL_MEM_READ_WRITE,sizeof(uint)*2,NULL,&error_code);

  _CHECKCL_ERR_CODE(error_code,"Failed to create is_changed_buf");

  do
  {
    is_changed = 0;

    _LOG_VAR(iteration_ct++);

    error_code = clEnqueueWriteBuffer(commands,is_changed_buf,CL_TRUE,
                                      0,sizeof(uint),&is_changed,0,NULL,NULL);

    _CHECKCL_ERR_CODE(error_code,"Failed to write to is_changed_buf");

    // Set the arguments to our compute kernel
    //
    a = 0 ;

    error_code  = 0;
    error_code  = clSetKernelArg(kernel, a++, sizeof(cl_mem), &m_cell_flag_img);
    error_code  = clSetKernelArg(kernel, a++, sizeof(cl_mem), &m_cell_pair_img);
    error_code |= clSetKernelArg(kernel, a++, sizeof(cl_mem), &cell_own_img);
    error_code |= clSetKernelArg(kernel, a++, sizeof(cl_mem), &cell_own_img);
    error_code |= clSetKernelArg(kernel, a++, sizeof(cell_coord_t), &x_min);
    error_code |= clSetKernelArg(kernel, a++, sizeof(cell_coord_t), &x_max);
    error_code |= clSetKernelArg(kernel, a++, sizeof(cell_coord_t), &y_min);
    error_code |= clSetKernelArg(kernel, a++, sizeof(cell_coord_t), &y_max);
    error_code |= clSetKernelArg(kernel, a++, sizeof(cl_mem), &is_changed_buf);

    _CHECKCL_ERR_CODE(error_code,"Failed to set args for dobfs_markowner_extrema kernel");

    error_code = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
                                        global, local, 0, NULL, NULL);

    _CHECKCL_ERR_CODE(error_code,"Failed to execute dobfs_markowner_extrema kernel");

    clFinish(commands);

    error_code = clEnqueueReadBuffer(commands,is_changed_buf,CL_TRUE,
                                     0,sizeof(uint),&is_changed,0,NULL,NULL);

  _LOG_VAR(is_changed);

    _CHECKCL_ERR_CODE(error_code,"Failed to read to is_changed_buf");

  }
  while(is_changed == 1);

  clReleaseMemObject(is_changed_buf);

  cellpair_array_t cell_own;

  cell_own.resize ( (boost::extents[1+sz[0]][1+sz[1]]));

  error_code = clEnqueueReadImage( commands, cell_own_img, CL_TRUE,
                                   cell_img_ogn,cell_img_rgn,0,0,
                                   cell_own.data(),0,NULL,NULL);

  clReleaseMemObject(cell_own_img);

  for(int y = 0 ;y<= sz[1];++y)
    log_range(cell_own[y].begin(),cell_own[y].end());

}

void connectCps (GridDataset::mscomplex_t *msgraph,
                 GridDataset::cellid_t c1,
                 GridDataset::cellid_t c2)
{
  if (GridDataset::s_getCellDim (c1) <GridDataset::s_getCellDim (c2))
    std::swap (c1,c2);

  if (GridDataset::s_getCellDim (c1) != GridDataset::s_getCellDim (c2) +1)
    throw std::logic_error ("must connect i,i+1 cp (or vice versa)");

  if (msgraph->m_id_cp_map.find (c1) == msgraph->m_id_cp_map.end())
    throw std::logic_error (_SSTR ("cell not in id_cp_map c1="<<c1));

  if (msgraph->m_id_cp_map.find (c2) == msgraph->m_id_cp_map.end())
    throw std::logic_error (_SSTR ("cell not in id_cp_map c2="<<c2));

  uint cp1_ind = msgraph->m_id_cp_map[c1];

  uint cp2_ind = msgraph->m_id_cp_map[c2];

  GridDataset::critpt_t *cp1 = msgraph->m_cps[cp1_ind];

  GridDataset::critpt_t *cp2 = msgraph->m_cps[cp2_ind];

  cp1->des.insert (cp2_ind);

  cp2->asc.insert (cp1_ind);
}

inline bool lowestPairableCoFacet
    (GridDataset *dataset,
     GridDataset::cellid_t cellId,
     GridDataset::cellid_t& pairid
     )
{
  typedef GridDataset::cellid_t id_type;

  id_type cofacets[20];
  bool    cofacet_usable[20];

  uint cofacet_count = dataset->getCellCofacets ( cellId,cofacets );

  bool isTrueBoundryCell = dataset->isTrueBoundryCell ( cellId ) ;

  // for each co facet
  for ( uint i = 0 ; i < cofacet_count ; i++ )
  {
    id_type facets[20];
    uint facet_count = dataset->getCellFacets ( cofacets[i],facets );

    cofacet_usable[i] = true;

    if ( isTrueBoundryCell &&
         !dataset->isTrueBoundryCell ( cofacets[i] ) )
    {
      cofacet_usable[i] = false;
      continue;
    }

    for ( uint j = 0 ; j < facet_count ; j++ )
    {
      if ( dataset->compareCells ( cellId,facets[j] ))
      {
        cofacet_usable[i] = false;
        break;
      }
    }
  }

  bool pairid_usable = false;

  for ( uint i =0 ; i < cofacet_count;i++ )
  {
    if ( cofacet_usable[i] == false )
      continue;

    if(pairid_usable == false)
    {
      pairid_usable = true;
      pairid = cofacets[i];
      continue;
    }

    if ( dataset->compareCells ( cofacets[i],pairid ) )
      pairid = cofacets[i];

  }
  return pairid_usable;
}


void track_gradient_tree_bfs
    (GridDataset *dataset,
     GridDataset::mscomplex_t *msgraph,
     GridDataset::cellid_t start_cellId,
     eGradDirection gradient_dir
     )
{
  typedef GridDataset::cellid_t id_type;

  static uint ( GridDataset::*getcets[2] ) ( id_type,id_type * ) const =
  {
    &GridDataset::getCellFacets,
    &GridDataset::getCellCofacets
  };

  std::queue<id_type> cell_queue;

  // mark here that that cellid has no parent.

  cell_queue.push ( start_cellId );

  while ( !cell_queue.empty() )
  {
    id_type top_cell = cell_queue.front();

    cell_queue.pop();

    id_type      cets[20];

    uint cet_ct = ( dataset->*getcets[gradient_dir] ) ( top_cell,cets );

    for ( uint i = 0 ; i < cet_ct ; i++ )
    {
      if ( dataset->isCellCritical ( cets[i] ) )
      {
        connectCps(msgraph,start_cellId,cets[i]);
      }
      else
      {
        if ( !dataset->isCellExterior ( cets[i] ) )
        {
          id_type next_cell = dataset->getCellPairId ( cets[i] );

          if ( dataset->getCellDim ( top_cell ) ==
               dataset->getCellDim ( next_cell ) &&
               next_cell != top_cell )
          {
            // mark here that the parent of next cell is top_cell
            cell_queue.push ( next_cell );
          }
        }
      }
    }
  }
}

GridDataset::GridDataset (const rect_t &r,const rect_t &e) :
    m_rect (r),m_ext_rect (e),m_ptcomp(this)
{

  // TODO: assert that the given rect is of even size..
  //       since each vertex is in the even positions
  //
}

void GridDataset::init()
{
  rect_size_t   s = m_ext_rect.size();

  m_vertex_fns.resize (boost::extents[1+s[0]/2][1+s[1]/2]);
  m_cell_flags.resize ( (boost::extents[1+s[0]][1+s[1]]));
  m_cell_pairs.resize ( (boost::extents[1+s[0]][1+s[1]]));

  for (int y = 0 ; y<=s[1];++y)
    for (int x = 0 ; x<=s[0];++x)
      m_cell_flags[x][y] = CELLFLAG_UNKNOWN;

  rect_point_t bl = m_ext_rect.bottom_left();

  m_vertex_fns.reindex (bl/2);

  m_cell_flags.reindex (bl);

  m_cell_pairs.reindex (bl);
}

void GridDataset::clear_graddata()
{
  m_vertex_fns.resize (boost::extents[0][0]);
  m_cell_flags.resize (boost::extents[0][0]);
  m_cell_pairs.resize (boost::extents[0][0]);
  m_critical_cells.clear();
}

GridDataset::cellid_t   GridDataset::getCellPairId (cellid_t c) const
{
  if (m_cell_flags (c) &CELLFLAG_PAIRED == 0)
    throw std::logic_error ("invalid pair requested");

  return m_cell_pairs (c);
}

bool GridDataset::compareCells( cellid_t c1,cellid_t  c2 ) const
{
  if(getCellDim(c1) == 0)
    return ptLt(c1,c2);

  cellid_t pts1[20];
  cellid_t pts2[20];

  uint pts1_ct = getCellPoints ( c1,pts1);
  uint pts2_ct = getCellPoints ( c2,pts2);

  std::sort ( pts1,pts1+pts1_ct,m_ptcomp );
  std::sort ( pts2,pts2+pts2_ct,m_ptcomp);

  return std::lexicographical_compare
      ( pts1,pts1+pts1_ct,pts2,pts2+pts2_ct,
        m_ptcomp );
}

GridDataset::cell_fn_t GridDataset::get_cell_fn (cellid_t c) const
{
  cell_fn_t  fn = 0.0;

  cellid_t pts[20];

  uint pts_ct = getCellPoints (c,pts);

  for (int j = 0 ; j <pts_ct ;++j)
    fn += m_vertex_fns (pts[j]/2);

  fn /= pts_ct;

  return fn;
}

void GridDataset::set_cell_fn (cellid_t c,cell_fn_t f)
{
  if (getCellDim (c) != 0)
    throw std::logic_error ("values only for vertices are specified");

  c[0] /=2;

  c[1] /=2;

  m_vertex_fns (c) = f;
}

uint GridDataset::getCellPoints (cellid_t c,cellid_t  *p) const
{
  switch (getCellDim (c))
  {
  case 0:
    p[0] = c;
    return 1;
  case 1:
    {
      cell_coord_t d0 = c[0]&0x01,d1 = c[1]&0x01;
      p[0] = cellid_t (c[0]+d0,c[1]+d1);
      p[1] = cellid_t (c[0]-d0,c[1]-d1);
    }

    return 2;
  case 2:
    p[0] = cellid_t (c[0]+1,c[1]+1);
    p[1] = cellid_t (c[0]+1,c[1]-1);
    p[2] = cellid_t (c[0]-1,c[1]-1);
    p[3] = cellid_t (c[0]-1,c[1]+1);
    return 4;
  default:
    throw std::logic_error ("impossible dim");
    return 0;
  }
}

uint GridDataset::getCellFacets (cellid_t c,cellid_t *f) const
{
  switch (getCellDim (c))
  {
  case 0:
    return 0;
  case 1:
    {
      cell_coord_t d0 = c[0]&0x01,d1 = c[1]&0x01;
      f[0] = cellid_t (c[0]+d0,c[1]+d1);
      f[1] = cellid_t (c[0]-d0,c[1]-d1);
    }

    return 2;
  case 2:
    f[0] = cellid_t (c[0]  ,c[1]+1);
    f[1] = cellid_t (c[0]  ,c[1]-1);
    f[2] = cellid_t (c[0]-1,c[1]);
    f[3] = cellid_t (c[0]+1,c[1]);
    return 4;
  default:
    throw std::logic_error ("impossible dim");
    return 0;
  }
}

uint GridDataset::getCellCofacets (cellid_t c,cellid_t *cf) const
{
  uint cf_ct = 0;

  switch (getCellDim (c))
  {
  case 0:
    cf[0] = cellid_t (c[0]  ,c[1]+1);
    cf[1] = cellid_t (c[0]  ,c[1]-1);
    cf[2] = cellid_t (c[0]-1,c[1]);
    cf[3] = cellid_t (c[0]+1,c[1]);
    cf_ct =  4;
    break;
  case 1:
    {
      cell_coord_t d0 = c[0]&0x01,d1 = c[1]&0x01;
      cf[0] = cellid_t (c[0]+d1,c[1]+d0);
      cf[1] = cellid_t (c[0]-d1,c[1]-d0);
      cf_ct =  2;
    }

    break;
  case 2:
    return 0;
  default:
    throw std::logic_error ("impossible dim");
    return 0;
  }

  // position in cf[] where the next valid cf should be placed
  uint cf_nv_pos = 0;

  for (uint i = 0 ;i < cf_ct;++i)
    if (m_ext_rect.contains (cf[i]))
      cf[cf_nv_pos++] = cf[i];

  return cf_nv_pos;

}

uint GridDataset::getMaxCellDim() const
{
  return 2;
}

bool GridDataset::isPairOrientationCorrect (cellid_t c, cellid_t p) const
{
  return (getCellDim (c) <getCellDim (p));
}

bool GridDataset::isCellMarked (cellid_t c) const
{
  return ! (m_cell_flags (c) == CELLFLAG_UNKNOWN);
}

bool GridDataset::isCellCritical (cellid_t c) const
{
  return (m_cell_flags (c) & CELLFLAG_CRITCAL);
}

bool GridDataset::isCellPaired (cellid_t c) const
{
  return (m_cell_flags (c) & CELLFLAG_PAIRED);
}

void GridDataset::pairCells (cellid_t c,cellid_t p)
{
  m_cell_pairs (c) = p;
  m_cell_pairs (p) = c;

  m_cell_flags (c) = m_cell_flags (c) |CELLFLAG_PAIRED;
  m_cell_flags (p) = m_cell_flags (p) |CELLFLAG_PAIRED;
}

void GridDataset::markCellCritical (cellid_t c)
{
  m_cell_flags (c) = m_cell_flags (c) |CELLFLAG_CRITCAL;
}

bool GridDataset::isTrueBoundryCell (cellid_t c) const
{
  return (m_ext_rect.isOnBoundry (c));
}

bool GridDataset::isFakeBoundryCell (cellid_t c) const
{
  return (m_rect.isOnBoundry (c) && (!m_ext_rect.isOnBoundry (c)));
}

bool GridDataset::isCellExterior (cellid_t c) const
{
  return (!m_rect.contains (c) && m_ext_rect.contains (c));
}

std::string GridDataset::getCellFunctionDescription (cellid_t c) const
{
  std::stringstream ss;

  ( (std::ostream &) ss) <<c;

  return ss.str();

}

std::string GridDataset::getCellDescription (cellid_t c) const
{

  std::stringstream ss;

  ( (std::ostream &) ss) <<c;

  return ss.str();

}

void  GridDataset::assignGradients()
{

  // determine all the pairings of all cells in m_rect
  for (cell_coord_t y = m_rect.bottom(); y <= m_rect.top();y += 1)
    for (cell_coord_t x = m_rect.left(); x <= m_rect.right();x += 1)
    {
    cellid_t c (x,y),p;

    if (isCellMarked (c))
      continue;

    if (lowestPairableCoFacet (this,c,p))
      pairCells (c,p);
  }

  for (cell_coord_t y = m_rect.bottom(); y <= m_rect.top();y += 1)
    for (cell_coord_t x = m_rect.left(); x <= m_rect.right();x += 1)
    {
    cellid_t c (x,y);

    if (!isCellMarked (c)) markCellCritical (c);
  }

  // mark artificial boundry as critical

  for (cell_coord_t x = m_rect.left(); x <= m_rect.right();x += 1)
  {
    cellid_t bcs[] = {cellid_t (x,m_rect.bottom()),cellid_t (x,m_rect.top()) };

    for (uint i = 0 ; i <sizeof (bcs) /sizeof (cellid_t);++i)
    {
      cellid_t &c = bcs[i];

      if (isCellCritical (c)) continue;

      cellid_t cf[20];

      u_int cf_ct =  getCellCofacets (c,cf);

      for (u_int j = 0 ; j <cf_ct;++j)
      {
        if (isCellExterior (cf[j]))
        {
          markCellCritical (c);
          markCellCritical (getCellPairId (c));
          break;
        }
      }
    }
  }

  for (cell_coord_t y = m_rect.bottom() +1; y < m_rect.top();y += 1)
  {
    cellid_t bcs[] = {cellid_t (m_rect.left(),y),cellid_t (m_rect.right(),y) };

    for (uint i = 0 ; i <sizeof (bcs) /sizeof (cellid_t);++i)
    {
      cellid_t &c = bcs[i];

      if (isCellCritical (c)) continue;

      cellid_t cf[20];

      u_int cf_ct =  getCellCofacets (c,cf);

      for (u_int j = 0 ; j <cf_ct;++j)
      {
        if (isCellExterior (cf[j]))
        {
          markCellCritical (c);
          markCellCritical (getCellPairId (c));
          break;
        }
      }
    }
  }

  collateCriticalPoints();
}

void  GridDataset::collateCriticalPoints()
{
  for (cell_coord_t y = m_ext_rect.bottom(); y <= m_ext_rect.top();y += 1)
    for (cell_coord_t x = m_ext_rect.left(); x <= m_ext_rect.right();x += 1)
    {
    cellid_t c (x,y);

    if (isCellCritical (c))
      m_critical_cells.push_back(c);
  }
}


inline void add_to_grad_tree_proxy (GridDataset::cellid_t)
{
  // just do nothing
  return;
}

void  GridDataset::computeConnectivity(mscomplex_t *msgraph)
{

  addCriticalPointsToMSComplex
      (msgraph,m_critical_cells.begin(),m_critical_cells.end());

  msgraph->m_cp_fns.resize(m_critical_cells.size());

  for (cellid_list_t::iterator it = m_critical_cells.begin() ;
  it != m_critical_cells.end();++it)
  {

    switch (getCellDim (*it))
    {
    case 0:
      track_gradient_tree_bfs(this,msgraph,*it,GRADIENT_DIR_UPWARD);
      break;
    case 2:
      track_gradient_tree_bfs(this,msgraph,*it,GRADIENT_DIR_DOWNWARD);
      break;
    default:
      break;
    }
  }

  for (cellid_list_t::iterator it = m_critical_cells.begin() ;
  it != m_critical_cells.end();++it)
  {
    cellid_t c = *it;

    uint cp_idx = msgraph->m_id_cp_map[c];

    msgraph->m_cp_fns[cp_idx] = get_cell_fn(c);

    if(!isCellPaired(c))  continue;

    msgraph->m_cps[cp_idx]->isBoundryCancelable = true;

    msgraph->m_cps[cp_idx]->pair_idx =
        msgraph->m_id_cp_map[getCellPairId(c)];
  }
}

void GridDataset::getCellCoord (cellid_t c,double &x,double &y,double &z)
{
  x = c[0];
  y = 0;
  z = c[1];

  cellid_t pts[20];

  if(m_ext_rect.contains(c))
  {
    y= get_cell_fn(c);

  }
}
