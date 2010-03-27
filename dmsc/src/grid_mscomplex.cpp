#include <cmath>
#include <queue>

#include <discreteMorseAlgorithm.h>
#include <grid_mscomplex.h>

void shallow_replicate_cp(GridMSComplex::mscomplex_t &msc,
                          const GridMSComplex::critpt_t &cp)
{
  if(msc.m_id_cp_map.count(cp.cellid) != 0)
    throw std::logic_error("this cp is present in msc");

  GridMSComplex::critpt_t * dest_cp = new GridMSComplex::critpt_t;

  dest_cp->isCancelled              = cp.isCancelled;
  dest_cp->isBoundryCancelable      = cp.isBoundryCancelable;
  dest_cp->isOnStrangulationPath    = cp.isOnStrangulationPath;
  dest_cp->cellid                   = cp.cellid;

  msc.m_id_cp_map[dest_cp->cellid]  = msc.m_cps.size();
  msc.m_cps.push_back(dest_cp);
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

      shallow_replicate_cp(*out_msc,*src_cp);
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

      if(!src_cp->isBoundryCancelable)
        continue;

      critpt_t * src_pair_cp = m_cps[src_cp->pair_idx];

      bool src_in_msc      = (msc->m_id_cp_map.count(src_cp->cellid) != 0);
      bool src_pair_in_msc = (msc->m_id_cp_map.count(src_pair_cp->cellid) != 0);

      if(!src_in_msc && !src_pair_in_msc)
        continue;

      if(!src_in_msc)
      {
        shallow_replicate_cp(*msc,*src_cp);
        msc->m_cp_fns.push_back(m_cp_fns[j]);
      }

      if(!src_pair_in_msc)
      {
        shallow_replicate_cp(*msc,*src_pair_cp);
        msc->m_cp_fns.push_back(m_cp_fns[src_cp->pair_idx]);
      }

      uint dest_cp_idx = msc->m_id_cp_map[src_cp->cellid];

      critpt_t *dest_cp = msc->m_cps[dest_cp_idx];

      if(!src_in_msc || !src_pair_in_msc || !dest_cp->isBoundryCancelable)
      {
        uint dest_pair_cp_idx  = msc->m_id_cp_map[src_pair_cp->cellid];
        critpt_t *dest_pair_cp = msc->m_cps[dest_pair_cp_idx];

        dest_cp->isBoundryCancelable      = true;
        dest_pair_cp->isBoundryCancelable = true;

        dest_cp->pair_idx      = dest_pair_cp_idx;
        dest_pair_cp->pair_idx = dest_cp_idx;
      }

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
            shallow_replicate_cp(*msc,*src_conn_cp);
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

struct persistence_comparator_t
{
  typedef std::pair<uint,uint> canc_pair_idx_t;

  typedef GridMSComplex::cell_fn_t cell_fn_t;
  typedef GridMSComplex::cellid_t  cellid_t;

  GridMSComplex *m_msc;

  persistence_comparator_t(GridMSComplex *m):m_msc(m){}

  bool operator()(const canc_pair_idx_t & p1, const canc_pair_idx_t &p2)
  {
    cell_fn_t f1 = m_msc->m_cp_fns[p1.first];
    cell_fn_t f2 = m_msc->m_cp_fns[p1.second];
    cell_fn_t f3 = m_msc->m_cp_fns[p2.first];
    cell_fn_t f4 = m_msc->m_cp_fns[p2.second];

    cell_fn_t d1 = std::abs(f2-f1);
    cell_fn_t d2 = std::abs(f4-f3);

    if(d1 != d2)
      return d1>d2;

    cellid_t c1 = m_msc->m_cps[p1.first]->cellid;
    cellid_t c2 = m_msc->m_cps[p1.second]->cellid;

    cellid_t c3 = m_msc->m_cps[p1.first]->cellid;
    cellid_t c4 = m_msc->m_cps[p1.second]->cellid;

    d1 = (c1[0]-c2[0])*(c1[0]-c2[0]) + (c1[1]-c2[1])*(c1[1]-c2[1]);
    d2 = (c3[0]-c4[0])*(c3[0]-c4[0]) + (c3[1]-c4[1])*(c3[1]-c4[1]);

    if(d1 != d2)
      return d1>d2;

    if(c1 > c2)
      std::swap(c1,c2);

    if(c3 > c4)
      std::swap(c3,c4);

    if(c1 != c3)
      return c1 > c3;

    return c2 > c4;
  }

};

void GridMSComplex::simplify(crit_idx_pair_list_t & canc_pairs_list,
                             double simplification_treshold)
{
  typedef std::priority_queue
      <crit_idx_pair_t,crit_idx_pair_list_t,persistence_comparator_t>
      canc_pair_priq_t;

  persistence_comparator_t comp(this);

  canc_pair_priq_t  canc_pair_priq(comp);

  // add every edge in the descending manifold of the critical point

  cell_fn_t max_persistence = 0.0;

  for(uint i = 0 ;i < m_cps.size();++i)
  {
    critpt_t *cp = m_cps[i];

    for(const_conn_iter_t it = cp->des.begin();it != cp->des.end() ;++it)
    {
      canc_pair_priq.push(std::make_pair(i,*it));

      cell_fn_t persistence = std::abs(m_cp_fns[i]-m_cp_fns[*it]);

      max_persistence = std::max(persistence,max_persistence );
    }
  }

  uint num_cancellations = 0;

  while (canc_pair_priq.size() !=0)
  {
    crit_idx_pair_t canc_pair = canc_pair_priq.top();

    canc_pair_priq.pop();

    uint v1 = canc_pair.first;
    uint v2 = canc_pair.second;

    critpt_t * cp1 = m_cps[v1];
    critpt_t * cp2 = m_cps[v2];

    cell_fn_t persistence = std::abs(m_cp_fns[v1]-m_cp_fns[v2]);

    if((double)persistence/(double)max_persistence > simplification_treshold)
      break;

    // pop the topmost item in the priority queue and cancel

    if (
        ( cp1->isCancelled           == false ) &&
        ( cp2->isCancelled           == false ) &&
        ( cp1->isOnStrangulationPath == false ) &&
        ( cp2->isOnStrangulationPath == false ) &&
        ( !m_rect.isOnBoundry(cp1->cellid)) &&
        ( !m_rect.isOnBoundry(cp2->cellid))
        )
    {
      cancelPairs ( this,v1,v2 );
      num_cancellations++;

      // by boundry cancelable I mean cancelable only ..:)
      cp1->isBoundryCancelable = true;
      cp2->isBoundryCancelable = true;

      cp1->pair_idx  = v2;
      cp2->pair_idx  = v1;

      canc_pairs_list.push_back(canc_pair);
    }
  }

  _LOG_VAR(num_cancellations);
}

void GridMSComplex::un_simplify(const crit_idx_pair_list_t &canc_pairs_list)
{
  for(crit_idx_pair_list_t::const_reverse_iterator it = canc_pairs_list.rbegin();
      it != canc_pairs_list.rend() ; ++it)
  {
    crit_idx_pair_t canc_pair = *it;

    uint v1 = canc_pair.first;
    uint v2 = canc_pair.second;

    uncancel_pairs(this,v1,v2);
  }
}

void GridMSComplex::simplify_un_simplify(double simplification_treshold)
{
  crit_idx_pair_list_t canc_pairs_list;

  simplify(canc_pairs_list,simplification_treshold);

  un_simplify(canc_pairs_list);
}


#include <fstream>

#define SAVE_TO_ASCII

void write_disc(const GridMSComplex::critpt_disc_t *disc,
                const std::string &prefix,
                const GridMSComplex::cellid_t &c )
{

  std::stringstream ss;
  ss<<prefix;
  ((std::ostream&)ss)<<c;

  std::ofstream os;

#ifdef SAVE_TO_ASCII
  os.open(ss.str().c_str(),std::ios::out|std::ios::ate|std::ios::app);
#else
  os.open(ss.str().c_str(),std::ios::out|std::ios::ate|std::ios::app|std::ios::binary);
#endif

  if(os.is_open() == false)
    throw std::runtime_error("asc/des disc file not writeable");
#ifdef SAVE_TO_ASCII
  for(int i = 0 ; i < disc->size();++i)
  {
    os<<(*disc)[i];
  }
#else
  os.write((const char*)(const void*)disc->data(),disc->size()*sizeof(GridMSComplex::cellid_t));
#endif

  os.close();
}

void GridMSComplex::write_discs(const std::string &fn_prefix)
{
  critpt_t * cp ;

  for(uint i = 0 ; i < m_cps.size();++i)
  {
    cp = m_cps[i];
    if(cp->asc_disc.size() != 0 )
      write_disc(&cp->asc_disc,fn_prefix+"asc_",cp->cellid);

    if(cp->des_disc.size() != 0 )
      write_disc(&cp->des_disc,fn_prefix+"des_",cp->cellid);
  }
}

#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/binary_object.hpp>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <grid_dataset.h>

namespace boost
{
  namespace serialization
  {
    template<class Archive>
    void serialize(Archive & ar, grid_types_t::rect_point_t & r, const unsigned int )
    {
      typedef boost::array<grid_types_t::rect_point_t::value_type,grid_types_t::rect_point_t::static_size> rect_point_base_t;

      ar & boost::serialization::base_object<rect_point_base_t>(r);
    }

    template<class Archive>
    void serialize(Archive & ar, grid_types_t::rect_t & r, const unsigned int )
    {
      ar & r.bl;
      ar & r.tr;
    }

    template<class Archive>
    void serialize(Archive & ar, GridMSComplex::critpt_t & c, const unsigned int )
    {
      ar & c.cellid;
      ar & c.asc;
      ar & c.des;
      ar & c.isBoundryCancelable;
      ar & c.isCancelled;
      ar & c.isOnStrangulationPath;
      ar & c.pair_idx;
    }


    template<class Archive>
    void serialize(Archive & ar, GridMSComplex & g, const unsigned int )
    {
      ar & g.m_rect;
      ar & g.m_ext_rect;
      ar & g.m_cp_fns;
      ar & g.m_id_cp_map;
      ar & g.m_cps;
    }

    template<class Archive>
    void serialize(Archive & ar, GridDataset & ds, const unsigned int )
    {
       ar & ds.m_rect;
       ar & ds.m_ext_rect;

       GridDataset::rect_size_t ext_sz = ds.m_ext_rect.size();
       uint num_data_items = (ext_sz[0]+1)*(ext_sz[1]+1);

       if(Archive::is_loading::value)
         ds.init(NULL);

       ar & make_binary_object(ds.m_cell_flags.data(),num_data_items*sizeof(GridDataset::cell_flag_t));
       ar & make_binary_object(ds.m_cell_own.data(),num_data_items*sizeof(GridDataset::cellid_t));
       ar & make_binary_object(ds.m_cell_pairs.data(),num_data_items*sizeof(GridDataset::cellid_t));
    }
  }
}

// without the explicit instantiations below, the program will
// fail to link for lack of instantiantiation of the above function
// The impls are visible only in this file to save compilation time..

template void boost::serialization::serialize<boost::archive::text_iarchive>(
    boost::archive::text_iarchive & ar,
    GridDataset & g,
    const unsigned int file_version
);

template void boost::serialization::serialize<boost::archive::text_oarchive>(
    boost::archive::text_oarchive & ar,
    GridDataset & g,
    const unsigned int file_version
);


// without the explicit instantiations below, the program will
// fail to link for lack of instantiantiation of the above function
// The impls are visible only in this file to save compilation time..

template void boost::serialization::serialize<boost::archive::text_iarchive>(
    boost::archive::text_iarchive & ar,
    GridMSComplex & g,
    const unsigned int file_version
);
template void boost::serialization::serialize<boost::archive::text_oarchive>(
    boost::archive::text_oarchive & ar,
    GridMSComplex & g,
    const unsigned int file_version
);


