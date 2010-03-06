/***************************************************************************
 *   Copyright (C) 2009 by nithin,,,   *
 *   nithin@gauss   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
#ifndef __RECTANGLE_COMPLEX_H__
#define __RECTANGLE_COMPLEX_H__

#include <vector>
#include <algorithm>
#include <iostream>
#include <set>
#include <map>

#include <boost/bind.hpp>

#include <cpputils.h>
#include <logutil.h>


template <typename coord_type>
class rectangle_complex
{

  public:

    typedef unsigned int uint;

    struct size_def:public two_tuple_t<coord_type>
    {
      inline coord_type x() const {return (*this)[0];}
      inline coord_type y() const {return (*this)[1];}

      size_def(const coord_type& x,const coord_type& y)
          :two_tuple_t<coord_type>(x,y){}

      static size_def sub ( const size_def & s1,const size_def & s2 )
      {
        return size_def ( s1.x-s2.x,s1.y-s2.y() );
      }
    };

    struct point_def:public two_tuple_t<coord_type>
    {
      inline const coord_type& x() const {return (*this)[0];}
      inline const coord_type& y() const {return (*this)[1];}

      inline coord_type& x(){return (*this)[0];}
      inline coord_type& y(){return (*this)[1];}


      point_def ( const coord_type &x,const coord_type &y ) :
          two_tuple_t<coord_type>(x,y){}

      point_def () :two_tuple_t<coord_type>(-1,-1){}

      static point_def add ( const point_def & p,const size_def & s )
      {
        return point_def ( p.x()+s.x(),p.y()+s.y() );
      }

      static point_def sub ( const point_def & p,const size_def & s )
      {
        return point_def ( p.x()-s.x(),p.y()-s.y() );
      }

      static size_def sub ( const point_def & p1,const point_def & p2 )
      {
        return size_def ( p1.x()-p2.x(),p1.y()-p2.y() );
      }

      point_def add ( const size_def & s ) const
      {
        return add ( *this,s );
      }

      point_def sub ( const size_def & s )
      {
        return sub ( *this,s );
      }

      size_def sub ( const point_def & p )
      {
        return sub ( *this,p );
      }
      
      point_def operator/(coord_type s) const
      {
	return point_def((*this)[0]/s,(*this)[1]/s);
      }

    };

    struct rectangle_def
    {
      typedef rectangle_def _SELF_NAME;

      point_def bl;
      point_def tr;


      size_def size()
      {
        return tr.sub(bl);
      }


      rectangle_def
      (
        const coord_type & start_x,
        const coord_type & start_y,
        const coord_type & end_x,
        const coord_type & end_y,
        const coord_type & bias = 1
      )

      {

        bl.x() = std::min ( start_x*bias,end_x*bias );
        bl.y() = std::min ( start_y*bias,end_y*bias );

        tr.x() = std::max ( start_x*bias,end_x*bias );
        tr.y() = std::max ( start_y*bias,end_y*bias );

      }

      rectangle_def
      (
        const point_def &p1,
        const point_def &p2
      )
      {
        bl.x() = std::min ( p1.x(),p2.x() );
        bl.y() = std::min ( p1.y(),p2.y() );

        tr.x() = std::max ( p1.x(),p2.x() );
        tr.y() = std::max ( p1.y(),p2.y() );

      }

      bool isInInterior ( const point_def & p ) const
      {
        return
          (
            ( p.x() > bl.x() ) &&
            ( p.y() > bl.y() ) &&
            ( p.x() < tr.x() ) &&
            ( p.y() < tr.y() )
          );

      }

      bool contains ( const point_def & p ) const
      {
        return
          (
            ( p.x() >= bl.x() ) &&
            ( p.y() >= bl.y() ) &&
            ( p.x() <= tr.x() ) &&
            ( p.y() <= tr.y() )
          );
      }

      bool isOnBoundry ( const point_def & p ) const
      {
        return ( contains ( p ) && !isInInterior ( p ) );
      }

      bool contains ( const _SELF_NAME &rec ) const
      {
        return
          (
            ( bl.x() < rec.bl.x() ) &&
            ( bl.y() < rec.bl.y() ) &&
            ( tr.x() > rec.tr.x() ) &&
            ( tr.y() > rec.tr.y() )
          );
      }

      bool intersects ( const _SELF_NAME &rec ) const
      {
        return
          ! (
            ( ( tr.x() ) < ( rec.bl.x() ) ) ||
            ( ( rec.tr.x() ) < ( bl.x() ) ) ||
            ( ( tr.y() ) < ( rec.bl.y() ) ) ||
            ( ( rec.tr.y() ) < ( bl.y() ) )
          );
      }

      void mul ( const coord_type & b )
      {
        bl.x() *= b;
        bl.y() *= b;
        tr.x() *= b;
        tr.y() *= b;
      }

      void div ( const coord_type & b )
      {
        bl.x() /= b;
        bl.y() /= b;
        tr.x() /= b;
        tr.y() /= b;
      }

      void shrink ( const size_def &s )
      {
        bl = point_def::add ( bl,s );
        tr = point_def::sub ( tr,s );
      }

      void grow ( const size_def &s )
      {
        bl = point_def::sub ( bl,s );
        tr = point_def::add ( tr,s );
      }

      point_def mid() const
      {
        return point_def ( ( bl.x()+tr.x() ) /2, ( bl.y()+tr.y() ) /2 );
      }

      coord_type left() const
      {
        return bl.x();
      }

      coord_type right() const
      {
        return tr.x();
      }

      coord_type top() const
      {
        return tr.y();
      }

      coord_type bottom() const
      {
        return bl.y();
      }

      point_def bottom_left() const
      {
        return point_def ( left(), bottom() );
      }

      point_def top_left() const
      {
        return point_def ( left(),top() );
      }

      point_def top_right() const
      {
        return point_def ( right(),top() );
      }
      point_def bottom_right() const
      {
        return point_def ( right(),bottom() );
      }

      uint getNumVerts()
      {
        return ( tr.x-bl.x+1 ) * ( tr.y-bl.y+1 );
      }

      uint getNumQuads()
      {
        return ( tr.x-bl.x() ) * ( tr.y-bl.y() );
      }

      friend std::ostream& operator<< ( std::ostream& o, const rectangle_def& r )
      {
        return o<<"[ bl="<<r.bl<<" tr= "<<r.tr<<"]";
      }
    };


    typedef rectangle_complex _SELF_NAME;

    rectangle_def m_rec_def;

    std::vector<rectangle_def> m_outside_rects;

    std::vector<rectangle_def> m_inside_rects;

    std::vector<rectangle_def> m_inside_lines;

  public:
    rectangle_complex
    (
      coord_type  start_x,
      coord_type  start_y,
      coord_type  end_x,
      coord_type  end_y
    ) :
        m_rec_def ( start_x,start_y,end_x,end_y,2 )
    {
      init();
    }

    rectangle_complex ( rectangle_def rec )
        :m_rec_def ( rec )
    {
      m_rec_def.mul ( 2 );
      init();
    }

    void init()
    {
      rectangle_def r = m_rec_def;
      r.shrink ( size_def ( 1,1 ) );
      m_inside_rects.push_back ( r );

    }

    bool remove_region ( const rectangle_def &rec )
    {
      rectangle_def rec2 = rec;

      rec2.mul ( 2 );

      _LOG_V ( "removing rec="<<rec2<<" from "<<m_rec_def );

      // it must not intersect with any removed regions.. if it touches what happens ??
      int num = ( int ) count_if
                (
                  m_outside_rects.begin(),
                  m_outside_rects.end(),
                  boost::bind ( std::mem_fun ( &rectangle_def::intersects ),&rec2,_1 )
                );

      if ( num != 0 )
      {
        return false;
      }

      m_outside_rects.push_back ( rec2 );

      std::vector<rectangle_def> inside_rects;
      std::vector<rectangle_def> inside_lines;

      for ( typename std::vector<rectangle_def>::iterator inside_rects_it = m_inside_rects.begin();
            inside_rects_it != m_inside_rects.end();inside_rects_it++ )
      {
        _LOG_V ( "shattering rec="<<*inside_rects_it<<" with "<<rec2 );
        shatter_inside_rect ( rec2,*inside_rects_it,inside_rects,inside_lines );
      }

      for ( typename std::vector<rectangle_def>::iterator inside_lines_it = m_inside_lines.begin();
            inside_lines_it != m_inside_lines.end();inside_lines_it++ )
      {
        _LOG_V ( "shattering line="<<*inside_lines_it<<" with "<<rec2 );
        shatter_inside_line ( rec2,*inside_lines_it,inside_lines );
      }

      m_inside_rects.clear();
      m_inside_lines.clear();

      m_inside_rects.insert ( m_inside_rects.begin(),inside_rects.begin(),inside_rects.end() );
      m_inside_lines.insert ( m_inside_lines.begin(),inside_lines.begin(),inside_lines.end() );


      return true;
    }

    bool check_inside_outside_intersection_I
    (
      const rectangle_def &outrect,
      const rectangle_def &inrect,
      std::vector<rectangle_def> &insiderects
    )
    {
      // intersection type I : None

      if ( !outrect.intersects ( inrect ) )
      {
        insiderects.push_back ( inrect );
        _LOG_V ( "adding rect="<<inrect );
        return true;
      }

      return false;
    }

    bool check_inside_outside_intersection_II
    (
      const rectangle_def &outrect,
      const rectangle_def &inrect,
      std::vector<rectangle_def> &insiderects
    )
    {

      // intersection type II : neither contains a point of the other but they still intersect

      if ( ( inrect.top() > outrect.top() && inrect.bottom() < outrect.bottom() ) &&
           ( inrect.left() > outrect.left() && inrect.right() < outrect.right() ) )
      {
        insiderects.push_back ( rectangle_def ( inrect.bottom_left(),point_def ( inrect.right(),outrect.bottom()-1 ) ) );
        insiderects.push_back ( rectangle_def ( inrect.top_left(),point_def ( inrect.right(),outrect.top() +1 ) ) );

        _LOG_V ( "added inside rect="<<* ( insiderects.end()-2 ) );
        _LOG_V ( "added inside rect="<<* ( insiderects.end()-1 ) );

        return true;
      }

      if ( ( inrect.top() < outrect.top() && inrect.bottom() > outrect.bottom() ) &&
           ( inrect.left() < outrect.left() && inrect.right() > outrect.right() ) )
      {
        insiderects.push_back ( rectangle_def ( inrect.bottom_left(),point_def ( outrect.left()-1,inrect.top() ) ) );
        insiderects.push_back ( rectangle_def ( inrect.top_right(),point_def ( outrect.right() +1,inrect.bottom() ) ) );

        _LOG_V ( "added inside rect="<<* ( insiderects.end()-2 ) );
        _LOG_V ( "added inside rect="<<* ( insiderects.end()-1 ) );

        return true;
      }

      return false;
    }

    bool check_inside_outside_intersection_III
    (
      const rectangle_def &outrect,
      const rectangle_def &inrect,
      std::vector<rectangle_def> &insiderects
    )
    {
      // intersection type III : outrect contains 2 of incrects points

      point_def  p[] = {outrect.bottom_left(),outrect.bottom_right(),outrect.top_right(),outrect.top_left() };
      point_def  p_[] = {inrect.bottom_left(),inrect.bottom_right(),inrect.top_right(),inrect.top_left() };

      for ( int i =0 ; i < 4 ;i++ )
      {
//             point_def &a = p[ ( i+0 ) %4];
//             point_def &b = p[ ( i+1 ) %4];
        point_def &c = p[ ( i+2 ) %4];
//             point_def &d = p[ ( i+3 ) %4];

        point_def &a_ = p_[ ( i+0 ) %4];
        point_def &b_ = p_[ ( i+1 ) %4];
        point_def &c_ = p_[ ( i+2 ) %4];
        point_def &d_ = p_[ ( i+3 ) %4];

        if ( outrect.contains ( a_ ) && outrect.contains ( d_ ) )
        {
          point_def  p ( 0,0 );

          switch ( i )
          {
            case 0:p.x() = c.x() + 1; p.y() = c_.y()    ;break;
            case 1:p.x() = c_.x()   ; p.y() = c.y() + 1 ;break;
            case 2:p.x() = c.x() - 1; p.y() = c_.y()    ;break;
            case 3:p.x() = c_.x()   ; p.y() = c.y() - 1 ;break;
          }

          insiderects.push_back ( rectangle_def ( p,b_ ) );
          return true;

        }
      }
      return false;

    }

    bool check_inside_outside_intersection_IV
    (
      const rectangle_def &outrect,
      const rectangle_def &inrect,
      std::vector<rectangle_def> &insiderects,
      std::vector<rectangle_def> &insidelines
    )
    {
      // intersection type IV : inrect and outrect contain a point of each other

      point_def  p[] = {outrect.bottom_left(),outrect.bottom_right(),outrect.top_right(),outrect.top_left() };
      point_def  p_[] = {inrect.bottom_left(),inrect.bottom_right(),inrect.top_right(),inrect.top_left() };

      for ( int i =0 ; i < 4 ;i++ )
      {
//             point_def &a = p[ ( i+0 ) %4];
//             point_def &b = p[ ( i+1 ) %4];
        point_def &c = p[ ( i+2 ) %4];
//             point_def &d = p[ ( i+3 ) %4];

        point_def &a_ = p_[ ( i+0 ) %4];
        point_def &b_ = p_[ ( i+1 ) %4];
        point_def &c_ = p_[ ( i+2 ) %4];
        point_def &d_ = p_[ ( i+3 ) %4];

        if ( inrect.contains ( c ) && outrect.contains ( a_ ) )
        {
          point_def  p ( 0,0 );

          switch ( i )
          {
            case 0:p.x() = c.x() + 1; p.y() = c.y() + 1 ;break;
            case 1:p.x() = c.x() - 1; p.y() = c.y() + 1 ;break;
            case 2:p.x() = c.x() - 1; p.y() = c.y() - 1 ;break;
            case 3:p.x() = c.x() + 1; p.y() = c.y() - 1 ;break;
          }
          insiderects.push_back ( rectangle_def ( p,c_ ) );

          switch ( i )
          {
            case 0:p.x() = c.x() - 1; p.y() = c.y() + 1 ;break;
            case 1:p.x() = c.x() + 1; p.y() = c.y() + 1 ;break;
            case 2:p.x() = c.x() + 1; p.y() = c.y() - 1 ;break;
            case 3:p.x() = c.x() - 1; p.y() = c.y() - 1 ;break;
          }
          insiderects.push_back ( rectangle_def ( p,d_ ) );

          switch ( i )
          {
            case 0:p.x() = c.x() + 1; p.y() = c.y() - 1 ;break;
            case 1:p.x() = c.x() - 1; p.y() = c.y() - 1 ;break;
            case 2:p.x() = c.x() - 1; p.y() = c.y() + 1 ;break;
            case 3:p.x() = c.x() + 1; p.y() = c.y() + 1 ;break;
          }
          insiderects.push_back ( rectangle_def ( p,b_ ) );

          switch ( i )
          {
            case 0:p.x() = c.x() + 1; p.y() = c.y();break;
            case 1:p.x() = c.x() - 1; p.y() = c.y();break;
            case 2:p.x() = c.x() - 1; p.y() = c.y();break;
            case 3:p.x() = c.x() + 1; p.y() = c.y();break;
          }
          insidelines.push_back ( rectangle_def ( p,point_def ( c.x(),c_.y() ) ) );

          switch ( i )
          {
            case 0:p.x() = c.x(); p.y() = c.y() + 1 ;break;
            case 1:p.x() = c.x(); p.y() = c.y() + 1 ;break;
            case 2:p.x() = c.x(); p.y() = c.y() - 1 ;break;
            case 3:p.x() = c.x(); p.y() = c.y() - 1 ;break;
          }
          insidelines.push_back ( rectangle_def ( p,point_def ( c_.x(),c.y() ) ) );
          return true;
        }
      }
      return false;
    }

    bool check_inside_outside_intersection_V
    (
      const rectangle_def &outrect,
      const rectangle_def &inrect,
      std::vector<rectangle_def> &insiderects,
      std::vector<rectangle_def> &insidelines
    )
    {
      // intersection type V : inrect contains outrect

      point_def  p[] = {outrect.bottom_left(),outrect.bottom_right(),outrect.top_right(),outrect.top_left() };
      point_def  p_[] = {inrect.bottom_left(),inrect.bottom_right(),inrect.top_right(),inrect.top_left() };

      if ( inrect.contains ( outrect ) )
      {
        point_def &a = p[0];
        point_def &b = p[1];
        point_def &c = p[2];
        point_def &d = p[3];

        point_def &a_ = p_[0];
        point_def &b_ = p_[1];
        point_def &c_ = p_[2];
        point_def &d_ = p_[3];

        insiderects.push_back ( rectangle_def ( a_,point_def ( b.x() - 1,b.y() - 1 ) ) );
        insiderects.push_back ( rectangle_def ( b_,point_def ( c.x() + 1,c.y() - 1 ) ) );
        insiderects.push_back ( rectangle_def ( c_,point_def ( d.x() + 1,d.y() + 1 ) ) );
        insiderects.push_back ( rectangle_def ( d_,point_def ( a.x() - 1,a.y() + 1 ) ) );

        _LOG_V ( "added inside rect="<<* ( insiderects.end()-4 ) );
        _LOG_V ( "added inside rect="<<* ( insiderects.end()-3 ) );
        _LOG_V ( "added inside rect="<<* ( insiderects.end()-2 ) );
        _LOG_V ( "added inside rect="<<* ( insiderects.end()-1 ) );

        insidelines.push_back ( rectangle_def ( point_def ( a.x() - 1,a.y() )    ,point_def ( a_.x(), a.y() ) ) );
        insidelines.push_back ( rectangle_def ( point_def ( b.x()    ,b.y() - 1 ),point_def ( b.x() , b_.y() ) ) );
        insidelines.push_back ( rectangle_def ( point_def ( c.x() + 1,c.y() )    ,point_def ( c_.x(), c.y() ) ) );
        insidelines.push_back ( rectangle_def ( point_def ( d.x()    ,d.y() + 1 ),point_def ( d.x() , d_.y() ) ) );

        _LOG_V ( "added inside line="<<* ( insidelines.end()-4 ) );
        _LOG_V ( "added inside line="<<* ( insidelines.end()-3 ) );
        _LOG_V ( "added inside line="<<* ( insidelines.end()-2 ) );
        _LOG_V ( "added inside line="<<* ( insidelines.end()-1 ) );

        return true;
      }

      return false;
    }

    bool check_inside_outside_intersection_VI
    (
      const rectangle_def &outrect,
      const rectangle_def &inrect,
      std::vector<rectangle_def> &insiderects,
      std::vector<rectangle_def> &insidelines
    )
    {
      // intersection type VI : intrect contains 2 of outrects points
      point_def  p[] = {outrect.bottom_left(),outrect.bottom_right(),outrect.top_right(),outrect.top_left() };
      point_def  p_[] = {inrect.bottom_left(),inrect.bottom_right(),inrect.top_right(),inrect.top_left() };

      for ( int i = 0; i < 4; i++ )
      {
//             point_def &a = p[ ( i+0 ) %4];
        point_def &b = p[ ( i+1 ) %4];
        point_def &c = p[ ( i+2 ) %4];
//             point_def &d = p[ ( i+3 ) %4];

        point_def &a_ = p_[ ( i+0 ) %4];
        point_def &b_ = p_[ ( i+1 ) %4];
        point_def &c_ = p_[ ( i+2 ) %4];
        point_def &d_ = p_[ ( i+3 ) %4];

        if ( inrect.contains ( c ) && inrect.contains ( b ) )
        {
          point_def  p1 ( 0,0 ),p2 ( 0,0 );

          switch ( i )
          {
            case 0:p1.x() = c_.x()   ; p1.y() = c.y() + 1;break;
            case 1:p1.x() = c.x() - 1; p1.y() = c_.y()   ;break;
            case 2:p1.x() = c_.x()   ; p1.y() = c.y() - 1;break;
            case 3:p1.x() = c.x() + 1; p1.y() = c_.y()   ;break;

          }
          insiderects.push_back ( rectangle_def ( p1,d_ ) );

          switch ( i )
          {
            case 0:p1.x() = c.x() + 1; p1.y() = c.y() - 1;break;
            case 1:p1.x() = c.x() + 1; p1.y() = c.y() + 1;break;
            case 2:p1.x() = c.x() - 1; p1.y() = c.y() + 1;break;
            case 3:p1.x() = c.x() - 1; p1.y() = c.y() - 1;break;

          }
          insiderects.push_back ( rectangle_def ( p1,b_ ) );

          switch ( i )
          {
            case 0:p1.x() = b.x() - 1; p1.y() = b.y() - 1;break;
            case 1:p1.x() = b.x() + 1; p1.y() = b.y() - 1;break;
            case 2:p1.x() = b.x() + 1; p1.y() = b.y() + 1;break;
            case 3:p1.x() = b.x() - 1; p1.y() = b.y() + 1;break;

          }
          insiderects.push_back ( rectangle_def ( p1,a_ ) );

          switch ( i )
          {
            case 0:p1 = c.add ( size_def ( 1,0 ) ); p2.x() = c_.x();p2.y()=c.y() ;break;
            case 1:p1 = c.add ( size_def ( 0,1 ) ); p2.x() = c.x() ;p2.y()=c_.y();break;
            case 2:p1 = c.sub ( size_def ( 1,0 ) ); p2.x() = c_.x();p2.y()=c.y() ;break;
            case 3:p1 = c.sub ( size_def ( 0,1 ) ); p2.x() = c.x() ;p2.y()=c_.y();break;
          }
          insidelines.push_back ( rectangle_def ( p1,p2 ) );

          switch ( i )
          {
            case 0:p1 = b.sub ( size_def ( 0,1 ) ); p2.x() = b.x() ;p2.y()=b_.y();break;
            case 1:p1 = b.add ( size_def ( 1,0 ) ); p2.x() = b_.x();p2.y()=b.y() ;break;
            case 2:p1 = b.add ( size_def ( 0,1 ) ); p2.x() = b.x() ;p2.y()=b_.y();break;
            case 3:p1 = b.sub ( size_def ( 1,0 ) ); p2.x() = b_.x();p2.y()=b.y() ;break;

          }
          insidelines.push_back ( rectangle_def ( p1,p2 ) );
          return true;
        }
      }
      return false;
    }

    void shatter_inside_rect
    (
      const rectangle_def &outrect,
      const rectangle_def &inrect,
      std::vector<rectangle_def> &insiderects,
      std::vector<rectangle_def> &insidelines
    )
    {
      if ( check_inside_outside_intersection_I ( outrect,inrect,insiderects ) ) return;
      if ( check_inside_outside_intersection_II ( outrect,inrect,insiderects ) ) return;
      if ( check_inside_outside_intersection_III ( outrect,inrect,insiderects ) ) return;
      if ( check_inside_outside_intersection_IV ( outrect,inrect,insiderects,insidelines ) ) return;
      if ( check_inside_outside_intersection_V ( outrect,inrect,insiderects,insidelines ) ) return;
      if ( check_inside_outside_intersection_VI ( outrect,inrect,insiderects,insidelines ) ) return;
    }

    void shatter_inside_line
    (
      const rectangle_def &outrect,
      const rectangle_def &insideline,
      std::vector<rectangle_def> &insidelines
    )
    {
      if ( check_inside_outside_intersection_I ( outrect,insideline,insidelines ) ) return;
      if ( check_inside_outside_intersection_II ( outrect,insideline,insidelines ) ) return;
      if ( check_inside_outside_intersection_III ( outrect,insideline,insidelines ) ) return;
    }

    /**
     *
     * Has a complexity of Q.log(V) + V.log(V) + No interior quads
     *
     * No of interior quads is usually quite small .. unless the complex is really awry
     *
     * @param vertlist
     * @param indlist
     * @param numverts
     * @param
     */
    void createVertAndQuadIndexLists
    (
      coord_type *&vertlist, // two coords per vertex
      uint       *&indlist, //four indices perquad
      uint        &numverts,
      uint        &numquads,
      uint        &numExtVerts,
      uint        &numExtQuads,
      const bool  &includeOutsideBoundryRect = true
    )
    {

      typedef std::map< std::pair<coord_type,coord_type>,uint,ordered_pair_comparator > point_map_t;
      typedef std::vector<point_def > point_list_t;

      point_list_t int_qlist;
      point_list_t ext_qlist;

      getInteriorQuads ( int_qlist );
      getExteriorQuadsForRemovedRegions ( ext_qlist );

      if ( includeOutsideBoundryRect == true )
        getExteriorQuadsForBoundingRect ( ext_qlist );

      point_map_t pmap;

      uint nextvertpos = 0;

      for ( typename point_list_t::iterator it = int_qlist.begin(); it != int_qlist.end() ;it++ )
      {
        if ( pmap.insert ( make_pair ( make_pair ( ( *it ).x()-1, ( *it ).y()-1 ),nextvertpos ) ).second == true ) ++nextvertpos;
        if ( pmap.insert ( make_pair ( make_pair ( ( *it ).x()-1, ( *it ).y()+1 ),nextvertpos ) ).second == true ) ++nextvertpos;
        if ( pmap.insert ( make_pair ( make_pair ( ( *it ).x()+1, ( *it ).y()+1 ),nextvertpos ) ).second == true ) ++nextvertpos;
        if ( pmap.insert ( make_pair ( make_pair ( ( *it ).x()+1, ( *it ).y()-1 ),nextvertpos ) ).second == true ) ++nextvertpos;
      }

      uint extVertBeginOffset = nextvertpos;

      for ( typename point_list_t::iterator it = ext_qlist.begin(); it != ext_qlist.end() ;it++ )
      {
        if ( pmap.insert ( make_pair ( make_pair ( ( *it ).x()-1, ( *it ).y()-1 ),nextvertpos ) ).second == true ) ++nextvertpos;
        if ( pmap.insert ( make_pair ( make_pair ( ( *it ).x()-1, ( *it ).y()+1 ),nextvertpos ) ).second == true ) ++nextvertpos;
        if ( pmap.insert ( make_pair ( make_pair ( ( *it ).x()+1, ( *it ).y()+1 ),nextvertpos ) ).second == true ) ++nextvertpos;
        if ( pmap.insert ( make_pair ( make_pair ( ( *it ).x()+1, ( *it ).y()-1 ),nextvertpos ) ).second == true ) ++nextvertpos;
      }

      numverts    = pmap.size();
      numquads    = int_qlist.size() + ext_qlist.size();
      numExtVerts = nextvertpos - extVertBeginOffset ;
      numExtQuads = ext_qlist.size();
      vertlist    = new coord_type[numverts *2];
      indlist     = new uint[numquads *4];

      for ( typename point_map_t::iterator it = pmap.begin(); it != pmap.end() ;it++ )
      {
        uint vertpos = ( *it ).second;
        vertlist[2*vertpos+0] = ( *it ).first.first/2;
        vertlist[2*vertpos+1] = ( *it ).first.second/2;
      }

      uint indpos = 0;

      for ( typename point_list_t::iterator it = int_qlist.begin(); it != int_qlist.end() ;it++ )
      {
        indlist[indpos+0] = pmap[make_pair ( ( *it ).x()-1, ( *it ).y()-1 ) ];
        indlist[indpos+1] = pmap[make_pair ( ( *it ).x()-1, ( *it ).y()+1 ) ];
        indlist[indpos+2] = pmap[make_pair ( ( *it ).x()+1, ( *it ).y()+1 ) ];
        indlist[indpos+3] = pmap[make_pair ( ( *it ).x()+1, ( *it ).y()-1 ) ];

        indpos+=4;
      }

      for ( typename point_list_t::iterator it = ext_qlist.begin(); it != ext_qlist.end() ;it++ )
      {
        indlist[indpos+0] = pmap[make_pair ( ( *it ).x()-1, ( *it ).y()-1 ) ];
        indlist[indpos+1] = pmap[make_pair ( ( *it ).x()-1, ( *it ).y()+1 ) ];
        indlist[indpos+2] = pmap[make_pair ( ( *it ).x()+1, ( *it ).y()+1 ) ];
        indlist[indpos+3] = pmap[make_pair ( ( *it ).x()+1, ( *it ).y()-1 ) ];

        indpos+=4;
      }

    }

    void getBoundryPoints ( const  rectangle_def &rec, const coord_type & point_step, std::vector<point_def > &qlist )
    {
      qlist.push_back ( rec.bottom_left() );

      for ( coord_type x = rec.left() + point_step ; x <=  rec.right()-point_step ; x += point_step )
      {
        qlist.push_back ( point_def ( x,rec.bottom() ) );
      }

      qlist.push_back ( rec.bottom_right() );

      for ( coord_type y = rec.bottom() + point_step ; y <= ( rec.top()-point_step ) ; y += point_step )
      {
        qlist.push_back ( point_def ( rec.right(),y ) );
      }

      qlist.push_back ( rec.top_right() );

      for ( coord_type x = rec.right() - point_step ; x >=  rec.left() +point_step ; x -= point_step )
      {
        qlist.push_back ( point_def ( x,rec.top() ) );
      }

      qlist.push_back ( rec.top_left() );

      for ( coord_type y = rec.top() - point_step ; y >= ( rec.bottom() +point_step ) ; y -= point_step )
      {
        qlist.push_back ( point_def ( rec.left(),y ) );
      }
    }


    void getExteriorQuadsForRemovedRegions ( std::vector<point_def > &qlist )
    {
      for ( uint i = 0 ; i < m_outside_rects.size();i++ )
      {
        rectangle_def outrect = m_outside_rects[i];

        outrect.shrink ( size_def ( 1,1 ) );

        getBoundryPoints ( outrect,2,qlist );
      }
    }

    void getExteriorQuadsForBoundingRect ( std::vector<point_def > &qlist )
    {
      rectangle_def outrect = m_rec_def;

      outrect.grow ( size_def ( 1,1 ) );

      getBoundryPoints ( outrect,2,qlist );
    }

    void getInteriorQuads ( std::vector<point_def > &qlist )
    {
      for ( coord_type i = 0 ; i < m_inside_rects.size();i++ )
      {
        rectangle_def rec = m_inside_rects[i];

        for ( coord_type x = rec.bl.x() ; x <= rec.tr.x(); x+=2 )
        {
          for ( coord_type y = rec.bl.y() ; y <= rec.tr.y(); y+=2 )
          {
            qlist.push_back ( point_def ( x,y ) );
          }
        }
      }
    }

};

#endif
