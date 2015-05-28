// Copyright (C) 2012-2014 Conrad Sanderson
// Copyright (C) 2012-2014 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup memory
//! @{

#include <fcntl.h> // for now; probably want to replace
#include <unistd.h> // same
#include <cerrno>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>

class memory
  {
  public:
  
                        arma_inline             static uword enlarge_to_mult_of_chunksize(const uword n_elem);
  
  template<typename eT>      inline arma_malloc static eT*   acquire(const uword n_elem, bool* mmap_happened);
  
  template<typename eT>      inline arma_malloc static eT*   acquire_chunked(const uword n_elem);
  
  template<typename eT> arma_inline             static void  release(eT* mem, size_t mlen, bool mapstate);
  
  
  template<typename eT> arma_inline static bool      is_aligned(const eT*  mem);
  template<typename eT> arma_inline static void mark_as_aligned(      eT*& mem);
  template<typename eT> arma_inline static void mark_as_aligned(const eT*& mem);
  };



arma_inline
uword
memory::enlarge_to_mult_of_chunksize(const uword n_elem)
  {
  const uword chunksize = arma_config::spmat_chunksize;
  
  // this relies on integer division
  const uword n_elem_mod = (n_elem > 0) ? (((n_elem-1) / chunksize) + 1) * chunksize : uword(0);
  
  return n_elem_mod;
  }



template<typename eT>
inline
arma_malloc
eT*
// changed function call to include mmap bit
memory::acquire(const uword n_elem, bool* mmap_happened=NULL) 
  {
  if (mmap_happened != NULL) { (*mmap_happened)=false;} // define mmap status
  eT* out_memptr;  
  arma_debug_check
    (
    ( size_t(n_elem) > (std::numeric_limits<size_t>::max() / sizeof(eT)) ),
    "arma::memory::acquire(): requested size is too large"
    );
  
  //eT* out_memptr;
  
  #if   defined(ARMA_USE_TBB_ALLOC)
    {
    out_memptr = (eT *) scalable_malloc(sizeof(eT)*n_elem);
    }
  #elif defined(ARMA_USE_MKL_ALLOC)
    {
    out_memptr = (eT *) mkl_malloc( sizeof(eT)*n_elem, 128 );
    }
  #elif defined(ARMA_HAVE_POSIX_MEMALIGN)
    {
    eT* memptr;
    
    const size_t alignment = 16;  // change the 16 to 64 if you wish to align to the cache line
    
    int status = posix_memalign((void **)&memptr, ( (alignment >= sizeof(void*)) ? alignment : sizeof(void*) ), sizeof(eT)*n_elem);
    
    out_memptr = (status == 0) ? memptr : NULL;
    }
  #elif defined(_MSC_VER)
    {
    out_memptr = (eT *) _aligned_malloc( sizeof(eT)*n_elem, 16 );  // lives in malloc.h
    }
  #else
    {
    //return ( new(std::nothrow) eT[n_elem] );
    out_memptr = (eT *) malloc(sizeof(eT)*n_elem);
    }
  #endif
   
  // TODO: for mingw, use __mingw_aligned_malloc
  
  if(n_elem > 0) // add mmap attempt into check of bad alloc
    {
    #if defined(ARMA_USE_MMAP)
      if (mmap_happened != NULL)
        {
          cout << "Mmap; better than bad, it's on!" << std::endl; // let me know when mmap has started for testing purposes
          char mmfp[] = "/mnt/ssd/tmp/armaMM.XXXXXX"; // unique filename; TODO add customizable path from build or something.
          int f = mkstemp(mmfp); // open the file.
          std::vector<eT> empty(10000000, 0); // make it bit by bit. Maybe try seekp if we can get it work without kernel truncating file
          std::fstream b(mmfp, std::ios_base::out | std::ios_base::app | std::ios_base::binary); // open file stream
            for(uword i = 0; i < ((n_elem/10000000)+1); i++){
              b.write((const char*) &empty[0], sizeof(eT)*10000000);
            }
              arma_debug_check
              (
              ( f == -1 ), // check if failed to write
              "arma::memory::acquire(): unable to create an empty binary"
              );
            int mmap_binary_default = open(mmfp, O_RDWR); // open file with readwrite permission so can operate.
            const size_t len = n_elem*sizeof(eT); // need this to unmap at least, also for arma's sake.
            out_memptr = (eT*) mmap(0, len, (PROT_READ|PROT_WRITE), MAP_SHARED, mmap_binary_default, 0); // give mmaped pointer
            arma_debug_check( ( out_memptr == MAP_FAILED ), strerror(errno) ); // check if mmap otherwise failed, and report why.
            b.close(); // close stream
            close(f); // close file
            if (mmap_happened !=NULL) {(*mmap_happened) = true;} // give us an indicator it worked for memtype and such
            //return out_memptr; // good job
          }
      #endif
      arma_check_bad_alloc( (out_memptr == NULL), "arma::memory::acquire(): out of memory" ); // check if anything else fell through the cracks.
      }
  return out_memptr;
  }



//! get memory in multiples of chunks, holding at least n_elem
template<typename eT>
inline
arma_malloc
eT*
memory::acquire_chunked(const uword n_elem)
  {
  const uword n_elem_mod = memory::enlarge_to_mult_of_chunksize(n_elem);
  
  return memory::acquire<eT>(n_elem_mod);
  }



template<typename eT>
arma_inline
void
memory::release(eT* mem, size_t mlen=0, bool mapstate=false) // give info to unmap, mapstate is memstate==4
  {
  #if   defined(ARMA_USE_TBB_ALLOC)
    {
    scalable_free( (void *)(mem) );
    }
  #elif defined(ARMA_USE_MKL_ALLOC)
    {
    mkl_free( (void *)(mem) );
    }
  #elif defined(ARMA_HAVE_POSIX_MEMALIGN)
    {
    free( (void *)(mem) );
    }
  #elif defined(_MSC_VER)
    {
    _aligned_free( (void *)(mem) );
    }
  #else
    {
    //delete [] mem;
    free( (void *)(mem) );
    }
  #endif
  
  // TODO: for mingw, use __mingw_aligned_free
  if (mapstate) // replaced a check if mmap is enabled
    {
    bool unmap_state=(int (munmap((void*)access::rw(mem),mlen)) == -1); // unmap it, essentially freeing it.
    arma_debug_check(unmap_state,"arma::memory::release(): unable to unmap matrix"); // tell me if something's wrong with release for mmap
    }
  else
    {
    arma_ignore(mlen);
    }
  }



template<typename eT>
arma_inline
bool
memory::is_aligned(const eT* mem)
  {
  #if (defined(ARMA_HAVE_ICC_ASSUME_ALIGNED) || defined(ARMA_HAVE_GCC_ASSUME_ALIGNED)) && !defined(ARMA_DONT_CHECK_ALIGNMENT)
    {
    return (sizeof(std::size_t) >= sizeof(eT*)) ? ((std::size_t(mem) & 0x0F) == 0) : false;
    }
  #else
    {
    arma_ignore(mem);
    
    return false;
    }
  #endif
  }



template<typename eT>
arma_inline
void
memory::mark_as_aligned(eT*& mem)
  {
  #if defined(ARMA_HAVE_ICC_ASSUME_ALIGNED)
    {
    __assume_aligned(mem, 16);
    }
  #elif defined(ARMA_HAVE_GCC_ASSUME_ALIGNED)
    {
    mem = (eT*)__builtin_assume_aligned(mem, 16);
    }
  #else
    {
    arma_ignore(mem);
    }
  #endif
  
  // TODO: MSVC?  __assume( (mem & 0x0F) == 0 );
  //
  // http://comments.gmane.org/gmane.comp.gcc.patches/239430
  // GCC __builtin_assume_aligned is similar to ICC's __assume_aligned,
  // so for lvalue first argument ICC's __assume_aligned can be emulated using
  // #define __assume_aligned(lvalueptr, align) lvalueptr = __builtin_assume_aligned (lvalueptr, align) 
  //
  // http://www.inf.ethz.ch/personal/markusp/teaching/263-2300-ETH-spring11/slides/class19.pdf
  // http://software.intel.com/sites/products/documentation/hpc/composerxe/en-us/cpp/lin/index.htm
  // http://d3f8ykwhia686p.cloudfront.net/1live/intel/CompilerAutovectorizationGuide.pdf
  }



template<typename eT>
arma_inline
void
memory::mark_as_aligned(const eT*& mem)
  {
  #if defined(ARMA_HAVE_ICC_ASSUME_ALIGNED)
    {
    __assume_aligned(mem, 16);
    }
  #elif defined(ARMA_HAVE_GCC_ASSUME_ALIGNED)
    {
    mem = (const eT*)__builtin_assume_aligned(mem, 16);
    }
  #else
    {
    arma_ignore(mem);
    }
  #endif
  }



//! @}
