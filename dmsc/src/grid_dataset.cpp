#include <grid_dataset.h>

#include <discreteMorseAlgorithm.h>
#include <vector>

#include <CL/cl.h>

#include <timer.h>

#include <QFile>

typedef GridDataset::cellid_t cellid_t;

cl_device_id s_device_id;             // compute device id
cl_context s_context;                 // compute context
cl_program s_grad_pgm;                 // compute program
cl_program s_coll_cps_pgm;                 // compute program


const int max_threads_1D   = 128;
const int max_threads_2D_x = 16;
const int max_threads_2D_y = 16;

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
  if (!s_context)
    throw std::runtime_error("Error: Failed to create a compute context!\n");

  // Create the gradient assignment compute program from the source buffer
  //

  QFile ocl_assign_grad_src ( ":/oclsources/assigngradient.cl" );
  ocl_assign_grad_src.open(QIODevice::ReadOnly);

  QByteArray assign_grad_src = ocl_assign_grad_src.readAll().constData();

  const char * assign_grad_chptr = assign_grad_src.constData();

  s_grad_pgm = clCreateProgramWithSource
               (s_context, 1, (const char **) & assign_grad_chptr,
                NULL, &error_code);

  if (!s_grad_pgm)
    throw std::runtime_error("Failed to create assign gradient program!\n");

  // Build the program executable
  //
  error_code = clBuildProgram(s_grad_pgm, 0, NULL, NULL, NULL, NULL);
  if (error_code != CL_SUCCESS)
  {
    size_t len;
    char buffer[2048];

    clGetProgramBuildInfo(s_grad_pgm, s_device_id, CL_PROGRAM_BUILD_LOG,
                          sizeof(buffer), buffer, &len);


    // Log the binary generated

    size_t bin_size;
    clGetProgramInfo(s_grad_pgm,CL_PROGRAM_BINARY_SIZES,
                     sizeof(size_t),&bin_size,NULL);

    char * ptx_buffer = new char[bin_size];

    clGetProgramInfo(s_grad_pgm,CL_PROGRAM_BINARIES,
                     sizeof(ptx_buffer),&ptx_buffer,NULL);

    _LOG_TO_FILE(std::string(ptx_buffer),"assigngradient.ptx");

    delete []ptx_buffer;

    throw std::runtime_error(std::string(buffer));
  }

  // Create the critical point collation compute program from the source buffer
  //

  QFile ocl_create_critpts_src ( ":/oclsources/collate_critpts.cl" );
  ocl_create_critpts_src.open(QIODevice::ReadOnly);

  QByteArray collate_critpts_src = ocl_create_critpts_src.readAll().constData();

  const char * collate_critpts_chptr = collate_critpts_src.constData();

  s_coll_cps_pgm
      = clCreateProgramWithSource(s_context, 1, & collate_critpts_chptr,
                                  NULL, &error_code);

  if (!s_coll_cps_pgm)
    throw std::runtime_error("Failed to create collate critpts program!\n");

  // Build the program executable
  //
  error_code = clBuildProgram(s_coll_cps_pgm, 0, NULL, NULL, NULL, NULL);
  if (error_code != CL_SUCCESS)
  {
    size_t len;
    char buffer[2048];

    clGetProgramBuildInfo(s_coll_cps_pgm, s_device_id,
                          CL_PROGRAM_BUILD_LOG,sizeof(buffer), buffer, &len);

    // Log the binary generated

    size_t bin_size;
    clGetProgramInfo(s_coll_cps_pgm,CL_PROGRAM_BINARY_SIZES,
                     sizeof(size_t),&bin_size,NULL);

    char * ptx_buffer = new char[bin_size];

    clGetProgramInfo(s_coll_cps_pgm,CL_PROGRAM_BINARIES,
                     sizeof(ptx_buffer),&ptx_buffer,NULL);

    _LOG_TO_FILE(std::string(ptx_buffer),"collatecritpts.ptx");

    delete []ptx_buffer;

    throw std::runtime_error(std::string(buffer));
  }
}

void GridDataset::stop_opencl()
{
  // Shutdown and cleanup
  //
  clReleaseProgram(s_coll_cps_pgm);
  clReleaseProgram(s_grad_pgm);
  clReleaseContext(s_context);

}


#define _GET_GLOBAL(s,l)\
(((s)/(2*(l)))*(2*(l)) + ((((s)%(2*(l))) == 0)?(0):(2*l)))

unsigned int nextPow2( unsigned int x )
{
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

void getNumBlocksAndThreads
    (int n, int maxThreads, size_t &blocks, size_t &threads)
{
  threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
  blocks = (n + threads - 1) / threads;
}

void GridDataset::assignGradients_ocl()
{

  Timer timer;

  timer.start();

  int error_code;                       // error code returned from api calls

  rect_size_t sz = m_ext_rect.size();

  size_t cell_img_ogn[3] = {0,0,0};
  size_t cell_img_rgn[3] = {sz[0]+1,sz[1]+1,1};

  size_t vert_img_rgn[3]  = {(sz[0]>>1)+1,(sz[1]>>1)+1,1};

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

  // Create a command commands
  //
  cl_command_queue commands = clCreateCommandQueue
                              (s_context, s_device_id, 0, &error_code);
  if (!commands)
    throw std::runtime_error("Error: Failed to create a command commands!\n");

  // Create the compute kernel in the program we wish to run
  //
  cl_kernel kernel = clCreateKernel(s_grad_pgm, "assign_gradient", &error_code);
  if (!kernel || error_code != CL_SUCCESS)
    throw std::runtime_error("Error: Failed to create compute kernel!\n");

  // Create the input and output arrays in device memory for our calculation
  //

  cl_image_format vert_fn_imgfmt,cell_pr_imgfmt,cell_fg_imgfmt;

  vert_fn_imgfmt.image_channel_data_type = CL_FLOAT;
  vert_fn_imgfmt.image_channel_order     = CL_R;

  cell_pr_imgfmt.image_channel_data_type = CL_SIGNED_INT16;
  cell_pr_imgfmt.image_channel_order     = CL_RG;

  cell_fg_imgfmt.image_channel_data_type = CL_UNSIGNED_INT8;
  cell_fg_imgfmt.image_channel_order     = CL_R;

  cl_mem vfn_img_cl = clCreateImage2D
                      (s_context,CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR,
                       &vert_fn_imgfmt,vert_img_rgn[0],vert_img_rgn[1],0,
                       m_vertex_fns.data(),&error_code);

  if (error_code != CL_SUCCESS)
    throw std::runtime_error("Failed to create cell fn device texture!\n");

  cl_mem cell_pr_img_cl = clCreateImage2D
                          (s_context,CL_MEM_WRITE_ONLY,
                           &cell_pr_imgfmt,cell_img_rgn[0],cell_img_rgn[1],0,
                           NULL,&error_code);

  if (error_code != CL_SUCCESS)
    throw std::runtime_error("Failed to create cell pair device texture!\n");

  cl_mem cell_fg_img_cl = clCreateImage2D
                          (s_context,CL_MEM_WRITE_ONLY,
                           &cell_fg_imgfmt,cell_img_rgn[0],cell_img_rgn[1],0,
                           NULL,&error_code);

  if (error_code != CL_SUCCESS)
    throw std::runtime_error("Failed to create cell flag device texture!\n");

  _LOG("Done tranfer Data    t = "<<timer.getElapsedTimeInMilliSec()<<" ms");

  if (!vfn_img_cl || !cell_pr_img_cl ||!cell_fg_img_cl)
    throw std::runtime_error("Error: Failed to allocate device memory!\n");


  cell_coord_t x_min = m_ext_rect.left()  ,x_max = m_ext_rect.right();
  cell_coord_t y_min = m_ext_rect.bottom(),y_max = m_ext_rect.top();

  // Set the arguments to our compute kernel
  //
  error_code = 0;
  error_code  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &vfn_img_cl);
  error_code |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &cell_pr_img_cl);
  error_code |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &cell_fg_img_cl);
  error_code |= clSetKernelArg(kernel, 3, sizeof(cell_coord_t), &x_min);
  error_code |= clSetKernelArg(kernel, 4, sizeof(cell_coord_t), &x_max);
  error_code |= clSetKernelArg(kernel, 5, sizeof(cell_coord_t), &y_min);
  error_code |= clSetKernelArg(kernel, 6, sizeof(cell_coord_t), &y_max);

  if (error_code != CL_SUCCESS)
    std::runtime_error("Error: Failed to set kernel arguments! \n");

  error_code = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
                                      global, local, 0, NULL, NULL);

  if (error_code)
    std::runtime_error("Error: Failed to execute kernel!\n");

  // Wait for the command commands to get serviced before launching next kernel
  //
  clFinish(commands);

  _LOG("Done gradient part 1 t = "<<timer.getElapsedTimeInMilliSec()<<" ms");

  // Release what we dont need anymore
  //
  clReleaseMemObject(vfn_img_cl);
  clReleaseKernel(kernel);

  kernel = clCreateKernel(s_grad_pgm, "complete_pairings", &error_code);
  if (!kernel || error_code != CL_SUCCESS)
    throw std::runtime_error("Error: Failed to create compute kernel!\n");

  // Set the arguments to our compute kernel
  //
  error_code = 0;
  error_code  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cell_pr_img_cl);
  error_code |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &cell_pr_img_cl);
  error_code |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &cell_fg_img_cl);
  error_code |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &cell_fg_img_cl);
  error_code |= clSetKernelArg(kernel, 4, sizeof(cell_coord_t), &x_min);
  error_code |= clSetKernelArg(kernel, 5, sizeof(cell_coord_t), &x_max);
  error_code |= clSetKernelArg(kernel, 6, sizeof(cell_coord_t), &y_min);
  error_code |= clSetKernelArg(kernel, 7, sizeof(cell_coord_t), &y_max);

  if (error_code != CL_SUCCESS)
    std::runtime_error("Error: Failed to set kernel arguments! \n");

  error_code = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
                                      global, local, 0, NULL, NULL);

  if (error_code)
    std::runtime_error("Error: Failed to execute kernel!\n");

  // Wait for the command commands to get serviced before copying results
  //
  clFinish(commands);

  _LOG("Done gradient part 2 t = "<<timer.getElapsedTimeInMilliSec()<<" ms");

  // Read back the results from the device to verify the output
  //

  error_code = clEnqueueReadImage( commands, cell_pr_img_cl, CL_TRUE,
                                   cell_img_ogn,cell_img_rgn,0,0,
                                   m_cell_pairs.data(),0,NULL,NULL);

  error_code = clEnqueueReadImage( commands, cell_fg_img_cl, CL_TRUE,
                                   cell_img_ogn,cell_img_rgn,0,0,
                                   m_cell_flags.data(),0,NULL,NULL);

  if (error_code != CL_SUCCESS)
    std::runtime_error("Error: Failed to read output images! \n");

  _LOG("Done reading results t = "<<timer.getElapsedTimeInMilliSec()<<" ms");

  // Release stuff we dont need anymore
  //

  clReleaseKernel(kernel);

  _LOG("Done grad assignment t = "<<timer.getElapsedTimeInMilliSec()<<" ms");

  // Figure out how we need to break up our collation work
  //

  size_t num_blocks[2];

  getNumBlocksAndThreads((sz[0]>>1)+1,max_threads_2D_x,num_blocks[0],local[0]);
  getNumBlocksAndThreads((sz[1]>>1)+1,max_threads_2D_y,num_blocks[1],local[1]);

  _LOG_VAR(num_blocks[0]);
  _LOG_VAR(num_blocks[1]);

  global[0] = num_blocks[0]*local[0];
  global[1] = num_blocks[1]*local[1];

  // Create the collation array to hold initial and intermediate results
  //

  typedef unsigned int counting_t;

  int critpt_ct_sz = num_blocks[0]*num_blocks[1]*sizeof(counting_t);

  _LOG_VAR(critpt_ct_sz);

  cl_mem critpt_ct_cl = clCreateBuffer
                        (s_context, CL_MEM_READ_WRITE,
                        critpt_ct_sz, NULL, &error_code);

  if (error_code != CL_SUCCESS || !critpt_ct_cl)
    throw std::runtime_error("Failed to create crit pt ct buffer!\n");

  cl_kernel collate_ker = clCreateKernel(s_coll_cps_pgm, "collate_cps_initcount", &error_code);
  if (!collate_ker || error_code != CL_SUCCESS)
    throw std::runtime_error("Failed creating collate_cps_initcount kernel!\n");

  // Set the arguments to our compute kernel
  //
  error_code = 0;
  error_code  = clSetKernelArg(collate_ker, 0, sizeof(cl_mem), &cell_fg_img_cl);
  error_code |= clSetKernelArg(collate_ker, 1, sizeof(cl_mem), &critpt_ct_cl);
  error_code |= clSetKernelArg(collate_ker, 2, sizeof(cell_coord_t), &x_min);
  error_code |= clSetKernelArg(collate_ker, 3, sizeof(cell_coord_t), &x_max);
  error_code |= clSetKernelArg(collate_ker, 4, sizeof(cell_coord_t), &y_min);
  error_code |= clSetKernelArg(collate_ker, 5, sizeof(cell_coord_t), &y_max);

  if (error_code != CL_SUCCESS)
    std::runtime_error("Error: Failed to set kernel arguments! \n");

  error_code = clEnqueueNDRangeKernel(commands, collate_ker, 2, NULL,
                                      global, local, 0, NULL, NULL);

  if (error_code != CL_SUCCESS)
    std::runtime_error("Error: Failed to execute kernel!\n");

  // Wait for the command commands to get serviced before copying results
  //
  clFinish(commands);

  counting_t critpt_collated_sz = critpt_ct_sz/sizeof(counting_t);

  while(critpt_collated_sz > 1)
  {
    size_t local,num_blocks;

    getNumBlocksAndThreads(critpt_collated_sz,max_threads_1D,num_blocks,local);

    size_t global = local*num_blocks;

    clReleaseKernel(collate_ker);

    collate_ker = clCreateKernel(s_coll_cps_pgm, "collate_cps_reduce", &error_code);
    if (!collate_ker || error_code != CL_SUCCESS)
      throw std::runtime_error("Failed creating collate_cps_initcount kernel!\n");

    error_code = 0;
    error_code  = clSetKernelArg(collate_ker, 0, sizeof(cl_mem), &critpt_ct_cl);
    error_code |= clSetKernelArg(collate_ker, 1, sizeof(counting_t), &critpt_collated_sz);

    if (error_code != CL_SUCCESS)
      std::runtime_error("Error: Failed to set kernel arguments! \n");

    error_code = clEnqueueNDRangeKernel(commands, collate_ker, 1, NULL,
                                        &global, &local, 0, NULL, NULL);

    if (error_code != CL_SUCCESS)
      std::runtime_error("Error: Failed to execute kernel!\n");

    // Wait for the command commands to get serviced before copying results
    //
    clFinish(commands);

    critpt_collated_sz = (critpt_collated_sz + (local*2-1)) / (local*2);

  }

  // Copy results
  //

  counting_t * reduce_res = new counting_t[critpt_ct_sz/sizeof(counting_t)];

  clEnqueueReadBuffer(commands, critpt_ct_cl, CL_TRUE, 0,
                      critpt_ct_sz,
                      reduce_res, 0, NULL, NULL);

  log_range(reduce_res,reduce_res + critpt_ct_sz/sizeof(counting_t),"reduce_res");

  clReleaseMemObject(critpt_ct_cl);
  clReleaseMemObject(cell_pr_img_cl);
  clReleaseMemObject(cell_fg_img_cl);
  clReleaseKernel(collate_ker);
  clReleaseCommandQueue(commands);

  // Validate our results
  //

  _LOG("Done grad assignment t = "<<timer.getElapsedTimeInMilliSec()<<" ms");

  collateCriticalPoints();

  collateCritcalPoints_ocl();
}


void GridDataset::collateCritcalPoints_ocl()
{

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
