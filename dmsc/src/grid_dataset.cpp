#include <grid_dataset.h>

#include <discreteMorseAlgorithm.h>

#include <vector>

typedef GridDataset::cellid_t cellid_t;

GridDataset::GridDataset (const rect_t &r,const rect_t &e) :
    m_rect (r),m_ext_rect (e)
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

bool GridDataset::ptLt (cellid_t c1,cellid_t c2) const
{
  double f1 = m_vertex_fns[c1[0]>>1][c1[1]>>1];
  double f2 = m_vertex_fns[c2[0]>>1][c2[1]>>1];

  if (f1 != f2)
    return f1 < f2;

  return c1<c2;

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
  cell_coord_t d0 = c[0]&0x01,d1 = c[1]&0x01;

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

  m_critical_cells.push_back (c);
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

    if (minPairable_cf (this,c,p))
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
}

inline void add_to_grad_tree_proxy (GridDataset::cellid_t)
{
  // just do nothing
  return;
}

void  GridDataset::computeConnectivity(mscomplex_t *msgraph)
{

  addCriticalPointsToMSComplex
      (msgraph,this,m_critical_cells.begin(),m_critical_cells.end());

  msgraph->m_cp_fns.resize(m_critical_cells.size());

  for (cellid_list_t::iterator it = m_critical_cells.begin() ;
  it != m_critical_cells.end();++it)
  {

    switch (getCellDim (*it))
    {
    case 0:
      track_gradient_tree_bfs
          (this,*it,GRADIENT_DIR_UPWARD,
           add_to_grad_tree_proxy,
           boost::bind (&connectCps,msgraph,*it,_1));
      break;
    case 2:
      track_gradient_tree_bfs
          (this,*it,GRADIENT_DIR_DOWNWARD,
           add_to_grad_tree_proxy,
           boost::bind (&connectCps,msgraph,*it,_1));
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

GridMSComplex::mscomplex_t * GridMSComplex::merge_up
    (const mscomplex_t& msc1,
     const mscomplex_t& msc2)
{

  // form the intersection rect
  rect_t ixn;

  if (!msc2.m_rect.intersection (msc1.m_rect,ixn))
    throw std::logic_error ("rects should intersect for merge");

  if ( (ixn.left() != ixn.right()) && (ixn.top() != ixn.bottom()))
    throw std::logic_error ("rects must merge along a 1 manifold");

  if (ixn.bottom_left() == ixn.top_right())
    throw std::logic_error ("rects cannot merge along a point alone");

  // TODO: ensure that the union of  rects is not including anything extra

  rect_t r =
      rect_t (std::min (msc1.m_rect.bottom_left(),msc2.m_rect.bottom_left()),
              std::max (msc1.m_rect.top_right(),msc2.m_rect.top_right()));

  rect_t e =
      rect_t (std::min (msc1.m_ext_rect.bottom_left(),msc2.m_ext_rect.bottom_left()),
              std::max (msc1.m_ext_rect.top_right(),msc2.m_ext_rect.top_right()));

  mscomplex_t * out_msc = new mscomplex_t(r,e);

  const mscomplex_t* msc_arr[] = {&msc1,&msc2};

  // make a union of the critical points in this
  for (uint i = 0 ; i <2;++i)
  {
    const mscomplex_t * msc = msc_arr[i];

    for (uint j = 0 ; j <msc->m_cps.size();++j)
    {
      const critpt_t *src_cp = msc->m_cps[j];

      // if it is contained or not
      if (i == 1 && (out_msc->m_id_cp_map.count(src_cp->cellid) == 1))
        continue;

      if(src_cp->isCancelled)
        continue;

      // allocate a new cp
      critpt_t* dest_cp = new critpt_t;

      // copy over the trivial data
      dest_cp->isCancelled                  = src_cp->isCancelled;
      dest_cp->isBoundryCancelable          = src_cp->isBoundryCancelable;
      dest_cp->isOnStrangulationPath        = src_cp->isOnStrangulationPath;
      dest_cp->cellid                       = src_cp->cellid;

      out_msc->m_id_cp_map[dest_cp->cellid] = out_msc->m_cps.size();

      out_msc->m_cps.push_back (dest_cp);
      out_msc->m_cp_fns.push_back(msc->m_cp_fns[j]);

    }
  }

  for (uint i = 0 ; i <2;++i)
  {
    const mscomplex_t * msc = msc_arr[i];

    // copy over connectivity information
    for (uint j = 0 ; j <msc->m_cps.size();++j)
    {
      const critpt_t *src_cp = msc->m_cps[j];

      if(src_cp->isCancelled)
        continue;

      critpt_t *dest_cp = out_msc->m_cps[out_msc->m_id_cp_map[src_cp->cellid]];

      if(src_cp->isBoundryCancelable)
      {
        critpt_t *src_pair_cp = msc->m_cps[src_cp->pair_idx];

        dest_cp->pair_idx = out_msc->m_id_cp_map[src_pair_cp->cellid];
      }

      const conn_t *acdc_src[]  = {&src_cp->asc, &src_cp->des};

      conn_t *acdc_dest[] = {&dest_cp->asc,&dest_cp->des};

      bool is_src_cmn_bndry = (ixn.contains(src_cp->cellid) && i == 1);

      for (uint j = 0 ; j < 2; ++j)
      {
        for (const_conn_iter_t it = acdc_src[j]->begin();
              it != acdc_src[j]->end();++it)
        {
          const critpt_t *conn_cp = msc->m_cps[*it];

          // common boundry connections would have been found along the boundry
          if( is_src_cmn_bndry && ixn.contains(conn_cp->cellid))
            continue;

          if (conn_cp->isCancelled)
            continue;

          acdc_dest[j]->insert (out_msc->m_id_cp_map[conn_cp->cellid]);
        }
      }
    }
  }

  // carry out the cancellation
  for(cell_coord_t y = ixn.bottom(); y <= ixn.top();++y)
  {
    for(cell_coord_t x = ixn.left(); x <= ixn.right();++x)
    {
      cellid_t c(x,y);

      if(out_msc->m_id_cp_map.count(c) != 1)
        throw std::logic_error("missing common bndry cp");

      u_int src_idx = out_msc->m_id_cp_map[c];

      critpt_t *src_cp = out_msc->m_cps[src_idx];

      if(src_cp->isCancelled || !src_cp->isBoundryCancelable)
        continue;

      u_int pair_idx = src_cp->pair_idx;

      cellid_t p = out_msc->m_cps[pair_idx]->cellid;

      if(!out_msc->m_rect.isInInterior(c)&& !out_msc->m_ext_rect.isOnBoundry(c))
        continue;

      if(!out_msc->m_rect.isInInterior(p)&& !out_msc->m_ext_rect.isOnBoundry(p))
        continue;

      cancelPairs(out_msc,src_idx,pair_idx);
    }
  }

  return out_msc;
}

void GridMSComplex::merge_down(mscomplex_t& msc1,mscomplex_t& msc2)
{
  // form the intersection rect
  rect_t ixn;

  if (!msc2.m_rect.intersection (msc1.m_rect,ixn))
    throw std::logic_error ("rects should intersect for merge");

  if ( (ixn.left() != ixn.right()) && (ixn.top() != ixn.bottom()))
    throw std::logic_error ("rects must merge along a 1 manifold");

  if (ixn.bottom_left() == ixn.top_right())
    throw std::logic_error ("rects cannot merge along a point alone");

  // carry out the uncancellation
  for(cell_coord_t y = ixn.top(); y >= ixn.bottom();--y)
  {
    for(cell_coord_t x = ixn.right(); x >= ixn.left();--x)
    {
      cellid_t c(x,y);

      if(this->m_id_cp_map.count(c) != 1)
        throw std::logic_error("missing common bndry cp");

      u_int src_idx = this->m_id_cp_map[c];

      critpt_t *src_cp = this->m_cps[src_idx];

      if(!src_cp->isCancelled )
        continue;

      u_int pair_idx = src_cp->pair_idx;

      cellid_t p = this->m_cps[pair_idx]->cellid;

      if(!this->m_rect.isInInterior(c)&& !this->m_ext_rect.isOnBoundry(c))
        continue;

      if(!this->m_rect.isInInterior(p)&& !this->m_ext_rect.isOnBoundry(p))
        continue;

      uncancel_pairs(this,src_idx,pair_idx);
    }
  }

  // identify and copy the results to msc1 and msc2

  mscomplex_t* msc_arr[] = {&msc1,&msc2};

  for (uint i = 0 ; i <2;++i)
  {
    mscomplex_t * msc = msc_arr[i];

    // adjust connections for uncancelled cps in msc
    for(uint j = 0 ; j < m_cps.size();++j)
    {
      critpt_t * src_cp = m_cps[j];

      if(src_cp->isCancelled)
        throw std::logic_error("all cps ought to be uncancelled by now");

      if(src_cp->isBoundryCancelable &&
         msc->m_id_cp_map.count(src_cp->cellid) == 0)
        continue;

      if(!src_cp->isBoundryCancelable)
        continue;

      if(msc->m_id_cp_map.count(src_cp->cellid) != 1)
        throw std::logic_error("this boundry cp must be contained in msc");

      critpt_t *dest_cp = msc->m_cps[msc->m_id_cp_map[src_cp->cellid]];

      conn_t *src_acdc[] = {&src_cp->asc,&src_cp->des};
      conn_t *dest_acdc[] = {&dest_cp->asc,&dest_cp->des};

      for(uint k = 0 ; k < 2;++k)
      {
        dest_acdc[k]->clear();

        for(conn_iter_t it = src_acdc[k]->begin(); it!=src_acdc[k]->end();++it)
        {
          critpt_t *src_conn_cp = m_cps[*it];

          if(src_conn_cp->isBoundryCancelable == true)
            throw std::logic_error("only non cancellable cps must be remaining");

          if(msc->m_id_cp_map.count(src_conn_cp->cellid) == 0)
          {
            critpt_t* dest_conn_cp = new critpt_t;

            // copy over the trivial data
            dest_conn_cp->isCancelled           = src_conn_cp->isCancelled;
            dest_conn_cp->isBoundryCancelable   = src_conn_cp->isBoundryCancelable;
            dest_conn_cp->isOnStrangulationPath = src_conn_cp->isOnStrangulationPath;
            dest_conn_cp->cellid                = src_conn_cp->cellid;

            msc->m_id_cp_map[dest_conn_cp->cellid] = msc->m_cps.size();
            msc->m_cps.push_back(dest_conn_cp);
            msc->m_cp_fns.push_back(m_cp_fns[*it]);
          }

          dest_acdc[k]->insert(msc->m_id_cp_map[src_conn_cp->cellid]);
        }// end it
      }// end k
    }// end j

    // adjust connections for non uncancelled cps in msc
    for(uint j = 0 ; j < m_cps.size();++j)
    {
      critpt_t * src_cp = m_cps[j];

      if(src_cp->isBoundryCancelable)
        continue;

      if(msc->m_id_cp_map.count(src_cp->cellid) != 1)
        continue;

      critpt_t *dest_cp = msc->m_cps[msc->m_id_cp_map[src_cp->cellid]];

      conn_t *src_acdc[] = {&src_cp->asc,&src_cp->des};
      conn_t *dest_acdc[] = {&dest_cp->asc,&dest_cp->des};

      for(uint k = 0 ; k < 2;++k)
      {
        dest_acdc[k]->clear();

        for(conn_iter_t it = src_acdc[k]->begin(); it!=src_acdc[k]->end();++it)
        {
          critpt_t *src_conn_cp = m_cps[*it];

          if(src_conn_cp->isBoundryCancelable == true)
          {
            _LOG("appears that "<<src_conn_cp->cellid<<"is still connected to"<<
                 src_cp->cellid);
            //throw std::logic_error("only non cancellable cps must be remaining 1");
          }

          if(msc->m_id_cp_map.count(src_conn_cp->cellid) == 0)
            continue;

          dest_acdc[k]->insert(msc->m_id_cp_map[src_conn_cp->cellid]);
        }// end it
      }// end k
    }// end j
  }//end i
}

void GridMSComplex::clear()
{
  std::for_each(m_cps.begin(),m_cps.end(),&delete_ftor<critpt_t>);
  m_cps.clear();
  m_id_cp_map.clear();
}


