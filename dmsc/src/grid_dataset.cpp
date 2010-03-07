#include <grid_dataset.h>

#include <discreteMorseAlgorithm.h>

#include <vector>

typedef GridDataset::cellid_t cellid_t;

GridDataset::GridDataset (const rect_t &r,const rect_t &e) :
    m_rect (r),m_ext_rect (e),m_msgraph(r,e)
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
  if (getCellDim (c) != 0)
    throw std::logic_error ("incorrect cell type requsting vert fn value");

  c[0] /=2;

  c[1] /=2;

  double ret = m_vertex_fns (c);

  return ret;
}

void GridDataset::set_cell_fn (cellid_t c,cell_fn_t f)
{
  if (getCellDim (c) != 0)
    throw std::logic_error ("incorrect cell type requsting vert fn value");

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

void GridDataset::connectCps (cellid_t c1, cellid_t c2)
{
  if (getCellDim (c1) <getCellDim (c2))
    std::swap (c1,c2);

  if (getCellDim (c1) != getCellDim (c2) +1)
    throw std::logic_error ("must connect i,i+1 cp (or vice versa)");

  if (m_msgraph.m_id_cp_map.find (c1) == m_msgraph.m_id_cp_map.end())
    throw std::logic_error (_SSTR ("cell not in id_cp_map c1="<<c1));

  if (m_msgraph.m_id_cp_map.find (c2) == m_msgraph.m_id_cp_map.end())
    throw std::logic_error (_SSTR ("cell not in id_cp_map c2="<<c2));

  uint cp1_ind = m_msgraph.m_id_cp_map[c1];

  uint cp2_ind = m_msgraph.m_id_cp_map[c2];

  critpt_t *cp1 = m_msgraph.m_cps[cp1_ind];

  critpt_t *cp2 = m_msgraph.m_cps[cp2_ind];

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

void  GridDataset::computeDiscs()
{

  addCriticalPointsToMSComplex
      (&m_msgraph,this,m_critical_cells.begin(),m_critical_cells.end());

  for (cellid_list_t::iterator it = m_critical_cells.begin() ;
  it != m_critical_cells.end();++it)
  {

    switch (getCellDim (*it))
    {
    case 0:
      track_gradient_tree_bfs
          (this,*it,GRADIENT_DIR_UPWARD,
           add_to_grad_tree_proxy,
           boost::bind (&GridDataset::connectCps,this,*it,_1));
      break;
    case 2:
      track_gradient_tree_bfs
          (this,*it,GRADIENT_DIR_DOWNWARD,
           add_to_grad_tree_proxy,
           boost::bind (&GridDataset::connectCps,this,_1,*it));
      break;
    default:
      break;
    }
  }

  for (cellid_list_t::iterator it = m_critical_cells.begin() ;
  it != m_critical_cells.end();++it)
  {
    cellid_t c = *it;

    uint cp_idx = m_msgraph.m_id_cp_map[c];

    if(!isCellPaired(c))  continue;

    m_msgraph.m_cps[cp_idx]->isBoundryCancelable = true;

    m_msgraph.m_cps[cp_idx]->pair_idx =
        m_msgraph.m_id_cp_map[getCellPairId(c)];
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
    uint pts_ct = getCellPoints (c,pts);

    for (int j = 0 ; j <pts_ct ;++j)
      y += get_cell_fn (pts[j]);

    y /= pts_ct;
  }
}

void GridMSComplex::merge_up (mscomplex_t &msc)
{

  // make a union of the critical points in this
  for (uint i = 0 ; i <msc.m_cps.size();++i)
  {
    critpt_t *src_cp = msc.m_cps[i];

    // if it is contained or not
    if (m_id_cp_map.count (src_cp->cellid) > 0)
      continue;

    // allocate a new cp
    critpt_t* dest_cp = new critpt_t;

    // copy over the trivial data
    dest_cp->isCancelled               = src_cp->isCancelled;
    dest_cp->isBoundryCancelable       = src_cp->isBoundryCancelable;
    dest_cp->isOnStrangulationPath     = src_cp->isOnStrangulationPath;
    dest_cp->cellid                    = src_cp->cellid;
    m_id_cp_map[dest_cp->cellid]       = m_cps.size();

    m_cps.push_back (dest_cp);

  }

  // form the intersection rect
  rect_t ixn(m_rect);

  if (!m_rect.intersection (msc.m_rect,ixn))
    throw std::logic_error ("rects should intersect for merge");

  if ( (ixn.left() != ixn.right()) && (ixn.top() != ixn.bottom()))
    throw std::logic_error ("rects must merge along a 1 manifold");

  if (ixn.bottom_left() == ixn.top_right())
    throw std::logic_error ("rects cannot merge along a point alone");

  // TODO: ensure that the uninon of  rects is not including anything extra

  // TODO: put away the old rects for unmerging

  m_ext_rect =
      rect_t (std::min (m_ext_rect.bottom_left(),msc.m_ext_rect.bottom_left()),
              std::max (m_ext_rect.top_right(),msc.m_ext_rect.top_right()));

  m_rect =
      rect_t (std::min (m_rect.bottom_left(),msc.m_rect.bottom_left()),
              std::max (m_rect.top_right(),msc.m_rect.top_right()));

  // copy over connectivity information
  for (uint i = 0 ; i <msc.m_cps.size();++i)
  {
    critpt_t *src_cp = msc.m_cps[i];

    if (src_cp->isCancelled)
      throw std::logic_error ("src cp should not yet be cancelled");

    if (m_id_cp_map.count (src_cp->cellid) != 1)
      throw std::logic_error ("missing cp in src");

    critpt_t *dest_cp = m_cps[m_id_cp_map[src_cp->cellid]];

    if(src_cp->isBoundryCancelable)
    {
      critpt_t *src_pair_cp = msc.m_cps[src_cp->pair_idx];

      if(src_pair_cp->pair_idx != i)
      {
        std::stringstream ss;

        ss<<"pairing is not reflected in the other"<<std::endl;
        ss<<"src_cp->cellid ="<<src_cp->cellid <<std::endl;
        ss<<"src_cp->pair_idx ="<<src_cp->pair_idx <<std::endl;
        ss<<"src_pair_cp->isCancelled ="<<src_pair_cp->isCancelled <<std::endl;
        ss<<"src_pair_cp->pair_idx = "<<src_pair_cp->pair_idx<<std::endl;
        ss<<"src_pair_cp->isBc="<<src_pair_cp->isBoundryCancelable <<std::endl;
        ss<<"i= "<<i<<std::endl;

        throw std::logic_error(ss.str());
      }

      if( !src_pair_cp->isBoundryCancelable )
        throw std::logic_error ("the other is not boundry cancelable");

      if (m_id_cp_map.count (src_pair_cp->cellid) != 1)
        throw std::logic_error ("missing cp src_pair_cp");

      dest_cp->pair_idx = m_id_cp_map[src_pair_cp->cellid];
    }

    conn_t *acdc_src[]  = {&src_cp->asc, &src_cp->des};

    conn_t *acdc_dest[] = {&dest_cp->asc,&dest_cp->des};

    bool is_src_cmn_bndry = ixn.contains(src_cp->cellid);

    for (uint j = 0 ; j < 2; ++j)
    {
      for (conn_iter_t it = acdc_src[j]->begin(); it != acdc_src[j]->end();++it)
      {
        critpt_t *conn_cp = msc.m_cps[*it];

        // common boundry connections would have been found along the boundry
        if( is_src_cmn_bndry && ixn.contains(conn_cp->cellid))
          continue;

        if (conn_cp->isCancelled)
          throw std::logic_error ("conn cp should not yet be cancelled");

        if (m_id_cp_map.count (conn_cp->cellid) != 1)
        {
          std::stringstream ss;
          ss<<"missing conn cp"<<conn_cp->cellid;
          throw std::logic_error (ss.str());
        }

        acdc_dest[j]->insert (m_id_cp_map[conn_cp->cellid]);
      }
    }
  }

  // carry out the cancellation
  for(cell_coord_t y = ixn.bottom(); y <= ixn.top();++y)
  {
    for(cell_coord_t x = ixn.left(); x <= ixn.right();++x)
    {
      cellid_t c(x,y);

      if(m_id_cp_map.count(c) != 1)
        throw std::logic_error("missing common bndry cp");

      u_int src_idx = m_id_cp_map[c];

      critpt_t *src_cp = m_cps[src_idx];

      if(src_cp->isCancelled || !src_cp->isBoundryCancelable)
        continue;

      u_int pair_idx = src_cp->pair_idx;

      cellid_t p = m_cps[pair_idx]->cellid;

      if(!m_rect.isInInterior(c)&& !m_ext_rect.isOnBoundry(c))
        continue;

      if(!m_rect.isInInterior(p)&& !m_ext_rect.isOnBoundry(p))
        continue;

      cancelPairs(this,src_idx,pair_idx);
    }
  }

  //copy the compacted resulting complex to msc

  std::for_each(msc.m_cps.begin(),msc.m_cps.end(),&delete_ftor<critpt_t>);
  msc.m_id_cp_map.clear();
  msc.m_cps.clear();

  msc.m_rect = m_rect;
  msc.m_ext_rect = m_ext_rect;

  for (uint i = 0 ; i <m_cps.size();++i)
  {
    critpt_t *src_cp = m_cps[i];

    if (src_cp->isCancelled)
      continue;

    // allocate a new cp
    critpt_t* dest_cp = new critpt_t;

    // copy over the trivial data
    dest_cp->isCancelled               = src_cp->isCancelled;
    dest_cp->isBoundryCancelable       = src_cp->isBoundryCancelable;
    dest_cp->isOnStrangulationPath     = src_cp->isOnStrangulationPath;
    dest_cp->cellid                    = src_cp->cellid;
    msc.m_id_cp_map[dest_cp->cellid]   = msc.m_cps.size();

    msc.m_cps.push_back (dest_cp);
  }

  // copy over connectivity information
  for (uint i = 0 ; i <m_cps.size();++i)
  {
    critpt_t *src_cp = m_cps[i];

    if (src_cp->isCancelled)
      continue;

    critpt_t *dest_cp = msc.m_cps[msc.m_id_cp_map[src_cp->cellid]];

    if(src_cp->isBoundryCancelable)
    {
      critpt_t *src_pair_cp = m_cps[src_cp->pair_idx];

      if(msc.m_id_cp_map.count(src_pair_cp->cellid) == 0)
      {

        critpt_t *src_pair_pair_cp = m_cps[src_pair_cp->pair_idx];
        critpt_t *src_pair_pair_pair_cp = m_cps[src_pair_pair_cp->pair_idx];

        _LOG_VAR(i);
        _LOG_VAR(src_cp->pair_idx);
        _LOG_VAR(src_pair_cp->pair_idx);
        _LOG_VAR(src_pair_pair_cp->pair_idx);
        _LOG_VAR(src_pair_pair_pair_cp->pair_idx);

        throw std::logic_error("missing pair");
      }

      dest_cp->pair_idx     = msc.m_id_cp_map[src_pair_cp->cellid];
    }

    conn_t *acdc_src[]  = {&src_cp->asc, &src_cp->des};

    conn_t *acdc_dest[] = {&dest_cp->asc,&dest_cp->des};

    for (uint j = 0 ; j < 2; ++j)
    {
      for (conn_iter_t it = acdc_src[j]->begin(); it != acdc_src[j]->end();++it)
      {
        critpt_t *conn_cp = m_cps[*it];

        if (conn_cp->isCancelled)
          continue;

        acdc_dest[j]->insert (msc.m_id_cp_map[conn_cp->cellid]);
      }
    }
  }
}
