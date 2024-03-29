!WRF:DRIVER_LAYER:IO
!
#define DEBUG_LVL 50
!#define mpi_x_comm_size(i,j,k)  Mpi_Comm_Size ( i,j,k ) ; write(0,*) __LINE__
#define mpi_x_comm_size(i,j,k)  Mpi_Comm_Size ( i,j,k )
! Workaround for bug in the IBM MPI implementation.  Look near the
! bottom of this file for an explanation.
#ifdef IBM_REDUCE_BUG_WORKAROUND
#define mpi_x_reduce(sb,rb,c,dt,op,r,com,ierr) reduce_add_integer(sb,rb,c,r,com)
#else
#define mpi_x_reduce(sb,rb,c,dt,op,r,com,ierr) MPI_Reduce(sb,rb,c,dt,op,r,com,ierr)
!#define mpi_x_reduce(sb,rb,c,dt,op,r,com,ierr) MPI_Reduce(sb,rb,c,dt,op,r,com,ierr) ; write(0,*)__LINE__
#endif

MODULE module_wrf_infer_cu
  USE module_internal_header_util
  USE module_timing
#if ( DA_CORE != 1 )
  USE module_cpl, ONLY : coupler_on, cpl_set_dm_communicator, cpl_finalize
#endif

  INTEGER, PARAMETER :: int_num_handles = 99
  INTEGER, PARAMETER :: exit_tag= 9999
  INTEGER, PARAMETER :: max_servers = int_num_handles+1  ! why +1?
  INTEGER, PARAMETER :: max_msg= int_num_handles+1  ! why +1?
  LOGICAL          :: infering_enabled
  LOGICAL          :: disable_infer= .FALSE.
  LOGICAL, EXTERNAL :: wrf_dm_on_monitor

  INTEGER  ninfer_groups
#ifdef DM_PARALLEL
  INTEGER :: mpi_comm_local_infer
  LOGICAL :: compute_node
  LOGICAL :: compute_group_master(max_servers)
  INTEGER :: mpi_comm_infer_groups(max_servers)
  INTEGER :: ninfer_tasks_per_group
  INTEGER :: ncompute_tasks
  INTEGER :: ntasks
  INTEGER :: mytask, req, res
  INTEGER :: msg_tag(max_msg)
#endif

  CONTAINS

#if  defined(DM_PARALLEL)  &&  !defined( STUBMPI )

    SUBROUTINE setup_infer_servers ( ninfer_tasks_per_group,     &
                                     n_groups_arg,            &
                                     mpi_comm_wrld,           &
                                     mpi_comm_local,          &
                                     mpi_comm_infer_groups)
      IMPLICIT NONE
      INCLUDE 'mpif.h'
      INTEGER,                      INTENT(IN)  :: ninfer_tasks_per_group, &
                                                   n_groups_arg, mpi_comm_wrld
      INTEGER,  INTENT(OUT)                     :: mpi_comm_local
      INTEGER, DIMENSION(100),      INTENT(OUT) :: mpi_comm_infer_groups
! Local
      INTEGER                     :: i, j, ii, comdup, ierr, ninfertasks, n_groups
      INTEGER, DIMENSION(ntasks)  :: icolor
      CHARACTER*255 mess
      INTEGER :: mytask, ntasks, nrank
      INTEGER :: me, ninfer, color, mpi_comm_node

      n_groups = n_groups_arg
      IF ( n_groups .LT. 1 ) n_groups = 1

      compute_node = .TRUE.
      call MPI_Comm_split_type(mpi_comm_wrld, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, mpi_comm_node,ierr )

      CALL MPI_Comm_rank(mpi_comm_wrld, mytask,ierr)
      CALL MPI_Comm_size(mpi_comm_node, ntasks,ierr)
      CALL MPI_Comm_rank(mpi_comm_node, nrank,ierr)

      ninfer = ninfer_tasks_per_group
      ncompute_tasks = ntasks - (ninfer * n_groups)

      IF ( ncompute_tasks .LT. ninfer ) THEN 
        WRITE(mess,'("Not enough tasks to have ",I3," groups of ",I3," Infer tasks. No infering.")')n_groups,ninfer
        ninfer = 0
        ncompute_tasks = ntasks
      ELSE                                   
        WRITE(mess,'("Infering with ",I3," groups of ",I3," Infer tasks.")')n_groups,ninfer
      ENDIF                                   
      CALL wrf_message(mess)

      IF ( ninfer .LE. 0 ) THEN
        infering_enabled = .FALSE.
        mpi_comm_local = mpi_comm_wrld
        mpi_comm_infer_groups = mpi_comm_wrld
        RETURN
      ENDIF

      infering_enabled = .TRUE.

! First construct the local communicators
! prepare to split the communicator by designating compute-only tasks
      color = 1
      if (nrank < ncompute_tasks) then
        color = 0
      endif

      CALL MPI_Comm_dup(mpi_comm_wrld,comdup,ierr)
      CALL MPI_Comm_split(comdup,color,mytask,mpi_comm_local,ierr)

      DO j = 1, n_groups
        ii = 1
        Do i =1, ntasks
          icolor(ii) = mod(i, ninfer*j)
          ii = ii + 1
        ENDDO

        CALL MPI_Comm_dup(mpi_comm_node,comdup,ierr)
        CALL MPI_Comm_split(comdup,icolor(nrank+1), mytask, &
                            mpi_comm_infer_groups(j),ierr)

        CALL MPI_Comm_size(mpi_comm_infer_groups(j), req,ierr)
        CALL MPI_Comm_rank(mpi_comm_infer_groups(j), res,ierr)
        WRITE(mess, *)'mzjinferserver3.0, color=',icolor(nrank+1),',io_groups=',mpi_comm_infer_groups(j),', mytask = ',&
        mytask,',nrank =',nrank, ', subcomm size = ',req, ', rank = ', res
        CALL wrf_message(mess)
      ENDDO

      compute_group_master = .FALSE.
      compute_node         = .FALSE.

      DO j = 1, n_groups
         IF ( nrank .LT. ncompute_tasks .OR.                                                  &    ! I am a compute task
              (ncompute_tasks+(j-1)*ninfer .LE. nrank .AND. nrank .LT. ncompute_tasks+j*ninfer)    &    ! I am the infer server for this group
            ) THEN

           ! Get the rank of this compute task in the compute+io 
           ! communicator to which it belongs
           CALL MPI_Comm_Rank( mpi_comm_infer_groups(j) , me , ierr )

           ! If I am an I/O server for this group then make that group's
           ! communicator the first element in the mpi_comm_io_groups array 
           ! (I will ignore all of the other elements).
           IF (ncompute_tasks+(j-1)*ninfer .LE. nrank .AND. nrank .LT. ncompute_tasks+j*ninfer) THEN
              mpi_comm_infer_groups(1) = mpi_comm_infer_groups(j)
           ELSE
            compute_node = .TRUE.
            ! If I am a compute task, check whether I am the member of my 
            ! group that will communicate things that should be sent just 
            ! once (e.g. commands) to the IO server of my group.
            compute_group_master(j) = (me .EQ. 0)

!           IF( compute_group_master(j) ) WRITE(*,*) mytask,': ARPDBG : I will talk to IO server in group ',j
          ENDIF
         ENDIF
      ENDDO

    END SUBROUTINE setup_infer_servers

    FUNCTION alloc_1d_buffer(n)
      IMPLICIT NONE
      INTEGER  :: n
      REAL, Allocatable, dimension(:) :: alloc_1d_buffer 
      allocate(alloc_1d_buffer(n))
      RETURN 

    END FUNCTION alloc_1d_buffer

    FUNCTION alloc_2d_buffer(shape_x, shape_y)
      IMPLICIT NONE
      INTEGER  :: shape_x, shape_y

      REAL, Allocatable, dimension(:,:) :: alloc_2d_buffer 
      allocate(alloc_2d_buffer(shape_x, shape_y))
      RETURN 

    END FUNCTION alloc_2d_buffer

    FUNCTION alloc_3d_buffer(shape_z, shape_x, shape_y)
      IMPLICIT NONE
      INTEGER  :: shape_z, shape_x, shape_y

      REAL, Allocatable, dimension(:,:,:) :: alloc_3d_buffer 
      allocate(alloc_3d_buffer(shape_z, shape_x, shape_y))
      RETURN 

    END FUNCTION alloc_3d_buffer

    SUBROUTINE infer_cu
      USE module_state_description
      USE module_quilt_outbuf_ops
      USE module_configure, only : grid_config_rec_type, model_config_rec, model_to_grid_config_rec
      USE module_cu_mskf_dl_replace
      IMPLICIT NONE
      INCLUDE 'mpif.h'
#include "intio_tags.h"
#include "wrf_io_flags.h"
      TYPE (grid_config_rec_type)  :: config_flags
      INTEGER  status(MPI_STATUS_SIZE)
      INTEGER itag, ninbuf, ntasks_infer_group, ntasks_local_group, mytask_local, ierr
      INTEGER istat
      INTEGER client, recv_size, tag
      INTEGER mytask_infer_group
      INTEGER, EXTERNAL :: use_package
      INTEGER i,iii, jjj, vid, CC, DD, dom_id
      LOGICAL           :: CALL_server_ready
      LOGICAL, EXTERNAL :: use_infer_servers

      character*120 sysline
      character*1000 message

      type ptrcontainer
        real, pointer :: ptr(:)
      end type ptrcontainer

      type(ptrcontainer), allocatable ::ptr_buf(:)

      INTEGER  shape_x, shape_y, shape_z, shape_z_lev
      INTEGER  last_shape_x, last_shape_y, last_shape_z, last_shape_z_lev
      INTEGER  size_p,  size_z, size_z_lev
      INTEGER  j, k

      REAL, allocatable, DIMENSION(:,:) :: hfx, &
                                           ust, &
                                           pblh
      REAL, allocatable, DIMENSION(:,:) :: pratec, & 
                                           nca                                          
      REAL, allocatable,  DIMENSION(:,:,:):: u, &
                                             v, &
                                             t, &
                                             qv, &
                                             p, &
                                             th, &
                                             dz8w, &
                                             rho, &
                                             pi
      REAL, allocatable, DIMENSION(:,:,:) :: w
      REAL, allocatable, DIMENSION(:,:,:) :: w0avg, & 
                                             rthcuten, &
                                             rqvcuten, &
                                             rqccuten, &
                                             rqrcuten, &
                                             rqicuten, &
                                             rqscuten 

      ! get info. about the I/O server group that this I/O server task
      ! belongs to
      ! Last task in this I/O server group is the I/O server "root"
      ! The I/O server "root" actually writes data to disk
      ! TBH:  WARNING:  This is also implicit in the CALL to collect_on_comm().
      CALL mpi_x_comm_size( mpi_comm_infer_groups(1), ntasks_infer_group,    ierr )
      CALL MPI_COMM_RANK( mpi_comm_infer_groups(1), mytask_infer_group,    ierr )
      CALL mpi_x_comm_size( mpi_comm_local_infer,        ntasks_local_group, ierr )
      CALL MPI_COMM_RANK( mpi_comm_local_infer,        mytask_local,       ierr )
      shape_x = 1
      shape_y = 1
      shape_z = 1
      shape_z_lev =1
      last_shape_x =1 
      last_shape_y = 1
      last_shape_z = 1
      last_shape_z_lev =1

      pratec = alloc_2d_buffer(shape_x, shape_y)
      nca = alloc_2d_buffer(shape_x, shape_y)
      hfx = alloc_2d_buffer(shape_x, shape_y)
      ust = alloc_2d_buffer(shape_x, shape_y)
      pblh= alloc_2d_buffer(shape_x, shape_y)

      u = alloc_3d_buffer(shape_z, shape_x, shape_y)
      v = alloc_3d_buffer(shape_z, shape_x, shape_y)
      t = alloc_3d_buffer(shape_z, shape_x, shape_y)
      qv = alloc_3d_buffer(shape_z, shape_x, shape_y)
      p = alloc_3d_buffer(shape_z, shape_x, shape_y)
      th = alloc_3d_buffer(shape_z, shape_x, shape_y)
      dz8w = alloc_3d_buffer(shape_z, shape_x, shape_y)
      rho = alloc_3d_buffer(shape_z, shape_x, shape_y)
      pi = alloc_3d_buffer(shape_z, shape_x, shape_y)

      w = alloc_3d_buffer(shape_z_lev, shape_x, shape_y)

      w0avg = alloc_3d_buffer(shape_z, shape_x, shape_y)
      rthcuten = alloc_3d_buffer(shape_z, shape_x, shape_y)
      rqvcuten = alloc_3d_buffer(shape_z, shape_x, shape_y)
      rqccuten = alloc_3d_buffer(shape_z, shape_x, shape_y)
      rqrcuten = alloc_3d_buffer(shape_z, shape_x, shape_y)
      rqicuten = alloc_3d_buffer(shape_z, shape_x, shape_y)
      rqscuten = alloc_3d_buffer(shape_z, shape_x, shape_y)

! Work out whether this i/o server processor has one fewer associated compute proc than
! the most any processor has. Can happen when number of i/o tasks does not evenly divide
! the number of compute tasks. This is needed to keep the i/o tasks sychronized on the
! same message when they start commmunicating to stitch together an output.
!
! Compute processes associated with this task:
       CC = ntasks_infer_group - 1
! Number of compute tasks per I/O task (less remainder)
       DD = ncompute_tasks / ntasks_local_group
!
       !CALL infer_init_cu(mytask_local)
       CALL infer_init_cu(-1)

! infinite loop until shutdown message received
! This is the main request-handling loop.  I/O quilt servers stay in this loop 
! until the model run ends.  
      DO WHILE (.TRUE.)  ! {
        CALL wrf_message("waiting and probe data")
      
        CALL MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG,mpi_comm_infer_groups(1),status, ierr)
        client = status(MPI_SOURCE)

        tag  = status(MPI_TAG)

        IF (tag .EQ. exit_tag)  then
            CALL MPI_Recv(shape_x, 1, MPI_INTEGER, client, exit_tag, mpi_comm_infer_groups(1), status, ierr)
            IF (coupler_on) THEN
               CALL cpl_finalize()
            ELSE
               CALL mpi_finalize(ierr)
            END IF
            STOP
        END IF

        write(message,*) 'probe data from client = ', client
        CALL wrf_debug(1, message)

        !write(message, *) 'after probe client =', client
        !CALL wrf_message(message)

        !send shape dims
        CALL MPI_Recv(shape_x, 1, MPI_INTEGER, client, msg_tag(1), mpi_comm_infer_groups(1), status, ierr)
        CALL MPI_Recv(shape_y, 1, MPI_INTEGER, client, msg_tag(2), mpi_comm_infer_groups(1), status, ierr)
        CALL MPI_Recv(shape_z, 1, MPI_INTEGER, client, msg_tag(3), mpi_comm_infer_groups(1), status, ierr)
        CALL MPI_Recv(shape_z_lev, 1, MPI_INTEGER, client, msg_tag(4), mpi_comm_infer_groups(1), status, ierr)

        !if (last_shape_x*last_shape_y .NE. shape_x*shape_y ) then
        if (last_shape_x .NE. shape_x           &
             .OR. last_shape_y .NE. shape_y     &
             .OR. last_shape_z .NE. shape_z     &
             .OR. last_shape_z_lev .NE. shape_z_lev) then

            write(message, *)'allocate shp_x =', shape_x, ', shp_y=', shape_y, ', shp_z=',shape_z, ',shp_z_lev =',shape_z_lev
            CALL wrf_debug(1, message)
            last_shape_x = shape_x
            last_shape_y = shape_y
            last_shape_z = shape_z
            last_shape_z_lev = shape_z_lev

            size_p = shape_x * shape_y 
            size_z = shape_z * shape_x * shape_y 
            size_z_lev = shape_z_lev * shape_x * shape_y 

            deallocate(pratec)
            deallocate(nca)
            deallocate(hfx)
            deallocate(ust)
            deallocate(pblh)
            
            deallocate(u)
            deallocate(v)
            deallocate(w)
            deallocate(t)
            deallocate(qv)
            deallocate(p)
            deallocate(th)
            deallocate(dz8w)
            deallocate(rho)
            deallocate(pi)
            
            deallocate(w0avg)
            deallocate(rthcuten)
            deallocate(rqvcuten)
            deallocate(rqccuten)
            deallocate(rqrcuten)
            deallocate(rqicuten)
            deallocate(rqscuten)

            pratec = alloc_2d_buffer(shape_x, shape_y)
            nca = alloc_2d_buffer(shape_x, shape_y)
            hfx = alloc_2d_buffer(shape_x, shape_y)
            ust = alloc_2d_buffer(shape_x, shape_y)
            pblh= alloc_2d_buffer(shape_x, shape_y)

            u = alloc_3d_buffer(shape_z, shape_x, shape_y)
            v = alloc_3d_buffer(shape_z, shape_x, shape_y)
            t = alloc_3d_buffer(shape_z, shape_x, shape_y)
            qv = alloc_3d_buffer(shape_z, shape_x, shape_y)
            p = alloc_3d_buffer(shape_z, shape_x, shape_y)
            th = alloc_3d_buffer(shape_z, shape_x, shape_y)
            dz8w = alloc_3d_buffer(shape_z, shape_x, shape_y)
            rho = alloc_3d_buffer(shape_z, shape_x, shape_y)
            pi = alloc_3d_buffer(shape_z, shape_x, shape_y)

            w = alloc_3d_buffer(shape_z_lev, shape_x, shape_y)

            w0avg = alloc_3d_buffer(shape_z, shape_x, shape_y)
            rthcuten = alloc_3d_buffer(shape_z, shape_x, shape_y)
            rqvcuten = alloc_3d_buffer(shape_z, shape_x, shape_y)
            rqccuten = alloc_3d_buffer(shape_z, shape_x, shape_y)
            rqrcuten = alloc_3d_buffer(shape_z, shape_x, shape_y)
            rqicuten = alloc_3d_buffer(shape_z, shape_x, shape_y)
            rqscuten = alloc_3d_buffer(shape_z, shape_x, shape_y)

        endif

        !CALL MPI_Probe(client, MPI_ANY_TAG,mpi_comm_infer_groups(1),status, ierr)
        !CALL MPI_Get_count(status, MPI_REAL, recv_size, ierr)

        !send 2d buffer
        CALL MPI_Recv(pratec(:,:),size_p, MPI_REAL, client, msg_tag(21), mpi_comm_infer_groups(1), ierr)
        CALL MPI_Recv(nca(:,:),  size_p, MPI_REAL, client, msg_tag(22), mpi_comm_infer_groups(1), ierr)
        CALL MPI_Recv(hfx(:,:),  size_p, MPI_REAL, client, msg_tag(23), mpi_comm_infer_groups(1), ierr)
        CALL MPI_Recv(ust(:,:),  size_p, MPI_REAL, client, msg_tag(24), mpi_comm_infer_groups(1), ierr)
        CALL MPI_Recv(pblh(:,:),  size_p, MPI_REAL, client, msg_tag(25), mpi_comm_infer_groups(1), ierr)

        ! send 3d shape_z
        CALL MPI_Recv(u(:,:,:),  size_z, MPI_REAL, client, msg_tag(41), mpi_comm_infer_groups(1), ierr)
        CALL MPI_Recv(v(:,:,:),  size_z, MPI_REAL, client, msg_tag(42), mpi_comm_infer_groups(1), ierr)
        CALL MPI_Recv(t(:,:,:),  size_z, MPI_REAL, client, msg_tag(43), mpi_comm_infer_groups(1), ierr)
        CALL MPI_Recv(qv(:,:,:),  size_z, MPI_REAL, client, msg_tag(44), mpi_comm_infer_groups(1), ierr)
        CALL MPI_Recv(p(:,:,:),  size_z, MPI_REAL, client, msg_tag(45), mpi_comm_infer_groups(1), ierr)
        CALL MPI_Recv(th(:,:,:),  size_z, MPI_REAL, client, msg_tag(46), mpi_comm_infer_groups(1), ierr)
        CALL MPI_Recv(dz8w(:,:,:),  size_z, MPI_REAL, client, msg_tag(47), mpi_comm_infer_groups(1), ierr)
        CALL MPI_Recv(rho(:,:,:),  size_z, MPI_REAL, client, msg_tag(48), mpi_comm_infer_groups(1), ierr)
        CALL MPI_Recv(pi(:,:,:),  size_z, MPI_REAL, client, msg_tag(49), mpi_comm_infer_groups(1), ierr)
        CALL MPI_Recv(w0avg(:,:,:),  size_z, MPI_REAL, client, msg_tag(50), mpi_comm_infer_groups(1), ierr)
        CALL MPI_Recv(rthcuten(:,:,:),  size_z, MPI_REAL, client, msg_tag(51), mpi_comm_infer_groups(1), ierr)
        CALL MPI_Recv(rqvcuten(:,:,:),  size_z, MPI_REAL, client, msg_tag(52), mpi_comm_infer_groups(1), ierr)
        CALL MPI_Recv(rqccuten(:,:,:),  size_z, MPI_REAL, client, msg_tag(53), mpi_comm_infer_groups(1), ierr)
        CALL MPI_Recv(rqrcuten(:,:,:),  size_z, MPI_REAL, client, msg_tag(54), mpi_comm_infer_groups(1), ierr)
        CALL MPI_Recv(rqicuten(:,:,:),  size_z, MPI_REAL, client, msg_tag(55), mpi_comm_infer_groups(1), ierr)
        CALL MPI_Recv(rqscuten(:,:,:),  size_z, MPI_REAL, client, msg_tag(56), mpi_comm_infer_groups(1), ierr)

        !send 3d shape_z_lev
        CALL MPI_Recv(w(:,:,:),  size_z_lev, MPI_REAL, client, msg_tag(61), mpi_comm_infer_groups(1), ierr)

        CALL infer_run_cu( &
                shape_x,            &
                shape_y,            &
                shape_z,            &
                shape_z_lev,        &
                pratec,             &
                nca,                &
                hfx,                &   
                ust,                &
                pblh,               &   
                u,                  &              
                v,                  &
                w,                  &
                t,                  &
                qv,                 &
                p,                  &
                th,                 &
                dz8w,               &
                rho,                &
                pi,                 &
                w0avg,              &
                rthcuten,           &
                rqvcuten,           &
                rqccuten,           &              
                rqrcuten,           &
                rqicuten,           &
                rqscuten)

      !CALL wrf_debug(1, "infer run cu over")

      CALL MPI_SSend(pratec(:,:),size_p, MPI_REAL, client, msg_tag(21), mpi_comm_infer_groups(1), ierr)
      CALL MPI_SSend(nca(:,:),  size_p, MPI_REAL, client, msg_tag(22), mpi_comm_infer_groups(1), ierr)

      CALL MPI_SSend(rthcuten(:,:,:),  size_z, MPI_REAL, client, msg_tag(51), mpi_comm_infer_groups(1), ierr)
      CALL MPI_SSend(rqvcuten(:,:,:),  size_z, MPI_REAL, client, msg_tag(52), mpi_comm_infer_groups(1), ierr)
      CALL MPI_SSend(rqccuten(:,:,:),  size_z, MPI_REAL, client, msg_tag(53), mpi_comm_infer_groups(1), ierr)
      CALL MPI_SSend(rqrcuten(:,:,:),  size_z, MPI_REAL, client, msg_tag(54), mpi_comm_infer_groups(1), ierr)
      CALL MPI_SSend(rqicuten(:,:,:),  size_z, MPI_REAL, client, msg_tag(55), mpi_comm_infer_groups(1), ierr)
      CALL MPI_SSend(rqscuten(:,:,:),  size_z, MPI_REAL, client, msg_tag(56), mpi_comm_infer_groups(1), ierr)

      END DO !}

    END SUBROUTINE infer_cu


! end of #endif of DM_PARALLEL
#endif

    SUBROUTINE init_module_wrf_infer_cu
      USE module_wrf_error, only: init_module_wrf_error
      USE module_driver_constants
      USE module_cu_mskf_dl_replace
#if defined( DM_PARALLEL ) && !defined( STUBMPI )
      USE module_dm, only: mpi_comm_allcompute
#endif
#if defined( DM_PARALLEL ) && !defined( STUBMPI )
      IMPLICIT NONE
      INCLUDE 'mpif.h'
      INTEGER i

      NAMELIST /namelist_infer/ ninfer_tasks_per_group, ninfer_groups
      INTEGER mytask_local, ntask_local
      !INTEGER ntasks, mytask, ierr, io_status
      INTEGER ierr, io_status
      INTEGER ntasks_infer_group, ntasks_local_group , mytask_infer_group

      INTEGER mpi_comm_here
      LOGICAL mpi_inited
      LOGICAL, EXTERNAL :: use_infer_servers
      character(100) :: message

!!!!! needed to sneak-peek the namelist to get parent_id
! define as temporaries
# include "namelist_defines.inc"

! Statements that specify the namelists
# include "namelist_statements.inc"
!TODO:  Change this to run-time switch

      infering_enabled = .FALSE.
      IF ( disable_infer ) RETURN

      CALL MPI_INITIALIZED( mpi_inited, ierr )
      IF ( .NOT. mpi_inited ) THEN
        IF ( coupler_on ) THEN
           CALL cpl_init( mpi_comm_here )
        ELSE
           CALL mpi_init ( ierr )
           mpi_comm_here = MPI_COMM_WORLD
        END IF
     
        CALL wrf_set_dm_communicator( mpi_comm_here )
        CALL wrf_set_dm_infer_comm( mpi_comm_here )   ! jm 20151212
        CALL wrf_termio_dup( mpi_comm_here )
      END IF

      CALL wrf_get_dm_infer_comm( mpi_comm_here )   ! jm 20151212

      CALL MPI_Comm_rank( mpi_comm_here, mytask, ierr ) ;
      CALL mpi_x_comm_size( mpi_comm_here, ntasks, ierr ) ;

      IF ( mytask .EQ. 0 ) THEN

        OPEN ( unit=27, file="namelist.input", form="formatted", status="old" )
        !added by mazhijian 
        ninfer_groups= 1
        ninfer_tasks_per_group = 0
        READ ( UNIT = 27 , NML = namelist_infer, IOSTAT=io_status )
        IF (io_status .NE. 0) THEN
          CALL wrf_error_fatal( "ERROR reading namelist namelist_inferXXXXX" )
        ENDIF

        REWIND(27)
        nproc_x = -1
        nproc_y = -1
        READ ( UNIT = 27 , NML = domains , IOSTAT=io_status )
        IF (io_status .NE. 0) THEN
          CALL wrf_error_fatal( "ERROR reading namelist domains" )
        ENDIF
        CLOSE ( 27 )

      ENDIF

      DO i=1, max_msg
         msg_tag(i) = i
      END DO

      CALL mpi_bcast( ninfer_tasks_per_group  , 1 , MPI_INTEGER , 0 , mpi_comm_here, ierr )
      CALL mpi_bcast( ninfer_groups , 1 , MPI_INTEGER , 0 , mpi_comm_here, ierr )

      CALL setup_infer_servers( ninfer_tasks_per_group,  &
                                ninfer_groups,           &
                                mpi_comm_here,        &
                                mpi_comm_local_infer,       &
                                mpi_comm_infer_groups)

      CALL mpi_x_comm_size( mpi_comm_infer_groups(1), ntasks_infer_group,    ierr )
      ! provide the communicator for the integration tasks to RSL
      write(message,*) ,'mzjinferserver1.0 ,mytask = ', mytask, ',compute_nod = ', compute_node, ', ntasks_infer_group = ',\
      ntasks_infer_group
      CALL wrf_message(message)
      IF ( compute_node ) THEN
#if defined( DM_PARALLEL ) && !defined( STUBMPI )
          mpi_comm_allcompute = mpi_comm_local_infer
          ! when server not supoort, use cpu infer 
          IF ( .NOT. use_infer_servers()) THEN
              call infer_init_cu(-1)
          END IF           
#endif
          CALL wrf_set_dm_communicator( mpi_comm_local_infer )
          CALL wrf_set_dm_quilt_comm( mpi_comm_local_infer )
#  if ( DA_CORE != 1 )
          IF (coupler_on)  THEN
          CALL cpl_set_dm_communicator( mpi_comm_local_infer )
           endif 
#  endif
       ELSE
#  if ( DA_CORE != 1 )
          IF (coupler_on) THEN
                 CALL cpl_set_dm_communicator( MPI_COMM_NULL )
           endif
#  endif
          CALL infer ! will not return on io server tasks
       ENDIF
#endif
      RETURN
    END SUBROUTINE init_module_wrf_infer_cu

    SUBROUTINE wrf_infer_run_cu( &
        shape_x,            &
        shape_y,            &
        shape_z,            &
        shape_z_lev,        &
        pratec,             &
        nca,                &
        hfx,                &   
        ust,                &
        pblh,               &   
        u,                  &              
        v,                  &
        w,                  &
        t,                  &
        qv,                 &
        p,                  &
        th,                 &
        dz8w,               &
        rho,                &
        pi,                 &
        w0avg,              &
        rthcuten,           &
        rqvcuten,           &
        rqccuten,           &              
        rqrcuten,           &
        rqicuten,           &
        rqscuten)

      USE module_cu_mskf_dl_replace
      INCLUDE 'mpif.h'
      INTEGER, INTENT(IN ) :: shape_x, shape_y, shape_z, shape_z_lev

      REAL, DIMENSION( shape_x, shape_y ), INTENT(IN)  :: hfx,    &
                                                          ust,    &
                                                          pblh

      REAL, DIMENSION(shape_z_lev, shape_x, shape_y), INTENT(IN) :: w
                                                 
      REAL, DIMENSION(shape_z, shape_x, shape_y), INTENT(IN) ::     u, &
                                                                    v, &
                                                                    t, &
                                                                    qv, &
                                                                    p, &
                                                                    th, &
                                                                    dz8w, &
                                                                    rho, &
                                                                    pi

      REAL, DIMENSION( shape_x, shape_y ), INTENT(INOUT)  :: pratec, &
                                                             nca    

      REAL, DIMENSION(shape_z, shape_x, shape_y), INTENT(INOUT) ::  w0avg, &
                                                                    rthcuten, &
                                                                    rqvcuten, &
                                                                    rqccuten, &
                                                                    rqrcuten, &
                                                                    rqicuten, &
                                                                    rqscuten


      INTEGER ntasks_group, mytask_group, server
      INTEGER size_p,  size_z, size_z_lev
      INTEGER x, y, z, z_lev
      INTEGER i, j, k
      LOGICAL, EXTERNAL :: use_infer_servers

      character(1000) :: message
      !INTEGER, EXTERNAL :: rsl_internal_microclock
      !INTEGER btimex_int_me
      !integer A
      !A = 0
      !btimex_int_me=rsl_internal_microclock()

      CALL mpi_x_comm_size( mpi_comm_infer_groups(1), server,    ierr )
      CALL mpi_comm_rank( MPI_COMM_WORLD, mm,    ierr )
      server = server -1

      IF ( .NOT. use_infer_servers()) THEN
        !write(message, *) ' my task = ', mm, ', server = ', server, ', shape_x= ',shape_x,', shape_y = ',shape_y, \
        !       ',shape_z =',shape_z, ', emsi(:,: ) = ', tlay(:3,:3,:3)
        !CALL wrf_message(message)

        CALL infer_run_cu( &
                shape_x,            &
                shape_y,            &
                shape_z,            &
                shape_z_lev,        &
                pratec,             &
                nca,                &
                hfx,                &   
                ust,                &
                pblh,               &   
                u,                  &              
                v,                  &
                w,                  &
                t,                  &
                qv,                 &
                p,                  &
                th,                 &
                dz8w,               &
                rho,                &
                pi,                 &
                w0avg,              &
                rthcuten,           &
                rqvcuten,           &
                rqccuten,           &              
                rqrcuten,           &
                rqicuten,           &
                rqscuten)
        
        !write(message, *) ' my task = ', mm, ', recv from server = ', server, ', rthcuten(:2,:2,:2) =', rthcuten(:2,:2,:2)
        !CALL wrf_message(message)

      else

      size_p = shape_x * shape_y
      size_z = shape_z * size_p
      size_z_lev = shape_z_lev * size_p

      CALL MPI_SSend(shape_x, 1, MPI_INTEGER, server, msg_tag(1), mpi_comm_infer_groups(1), ierr)
      CALL MPI_SSend(shape_y, 1, MPI_INTEGER, server, msg_tag(2), mpi_comm_infer_groups(1), ierr)
      CALL MPI_SSend(shape_z, 1, MPI_INTEGER, server, msg_tag(3), mpi_comm_infer_groups(1), ierr)
      CALL MPI_SSend(shape_z_lev, 1, MPI_INTEGER, server, msg_tag(4), mpi_comm_infer_groups(1), ierr)

      !send 2d buffer
      CALL MPI_SSend(pratec(:,:),size_p, MPI_REAL, server, msg_tag(21), mpi_comm_infer_groups(1), ierr)
      CALL MPI_SSend(nca(:,:),  size_p, MPI_REAL, server, msg_tag(22), mpi_comm_infer_groups(1), ierr)
      CALL MPI_SSend(hfx(:,:),  size_p, MPI_REAL, server, msg_tag(23), mpi_comm_infer_groups(1), ierr)
      CALL MPI_SSend(ust(:,:),  size_p, MPI_REAL, server, msg_tag(24), mpi_comm_infer_groups(1), ierr)
      CALL MPI_SSend(pblh(:,:),  size_p, MPI_REAL, server, msg_tag(25), mpi_comm_infer_groups(1), ierr)

      ! send 3d shape_z
      CALL MPI_SSend(u(:,:,:),  size_z, MPI_REAL, server, msg_tag(41), mpi_comm_infer_groups(1), ierr)
      CALL MPI_SSend(v(:,:,:),  size_z, MPI_REAL, server, msg_tag(42), mpi_comm_infer_groups(1), ierr)
      CALL MPI_SSend(t(:,:,:),  size_z, MPI_REAL, server, msg_tag(43), mpi_comm_infer_groups(1), ierr)
      CALL MPI_SSend(qv(:,:,:),  size_z, MPI_REAL, server, msg_tag(44), mpi_comm_infer_groups(1), ierr)
      CALL MPI_SSend(p(:,:,:),  size_z, MPI_REAL, server, msg_tag(45), mpi_comm_infer_groups(1), ierr)
      CALL MPI_SSend(th(:,:,:),  size_z, MPI_REAL, server, msg_tag(46), mpi_comm_infer_groups(1), ierr)
      CALL MPI_SSend(dz8w(:,:,:),  size_z, MPI_REAL, server, msg_tag(47), mpi_comm_infer_groups(1), ierr)
      CALL MPI_SSend(rho(:,:,:),  size_z, MPI_REAL, server, msg_tag(48), mpi_comm_infer_groups(1), ierr)
      CALL MPI_SSend(pi(:,:,:),  size_z, MPI_REAL, server, msg_tag(49), mpi_comm_infer_groups(1), ierr)
      CALL MPI_SSend(w0avg(:,:,:),  size_z, MPI_REAL, server, msg_tag(50), mpi_comm_infer_groups(1), ierr)
      CALL MPI_SSend(rthcuten(:,:,:),  size_z, MPI_REAL, server, msg_tag(51), mpi_comm_infer_groups(1), ierr)
      CALL MPI_SSend(rqvcuten(:,:,:),  size_z, MPI_REAL, server, msg_tag(52), mpi_comm_infer_groups(1), ierr)
      CALL MPI_SSend(rqccuten(:,:,:),  size_z, MPI_REAL, server, msg_tag(53), mpi_comm_infer_groups(1), ierr)
      CALL MPI_SSend(rqrcuten(:,:,:),  size_z, MPI_REAL, server, msg_tag(54), mpi_comm_infer_groups(1), ierr)
      CALL MPI_SSend(rqiuten(:,:,:),  size_z, MPI_REAL, server, msg_tag(55), mpi_comm_infer_groups(1), ierr)
      CALL MPI_SSend(rqscuten(:,:,:),  size_z, MPI_REAL, server, msg_tag(56), mpi_comm_infer_groups(1), ierr)

      !send 3d shape_z_lev
      CALL MPI_SSend(w(:,:,:),  size_z_lev, MPI_REAL, server, msg_tag(61), mpi_comm_infer_groups(1), ierr)

      !!write(message, *) ' my task = ', mm, ', send to server = ', server

      ! recv 2d buffer
      CALL MPI_Recv(pratec(:,:), size_p, MPI_REAL, server, msg_tag(71), mpi_comm_infer_groups(1), status, ierr)
      CALL MPI_Recv(nca(:,:), size_p, MPI_REAL, server, msg_tag(72), mpi_comm_infer_groups(1), status, ierr)

      ! recv 3d shape_z
      CALL MPI_Recv(w0avg(:,:,:), size_z, MPI_REAL, server, msg_tag(81), mpi_comm_infer_groups(1), status, ierr)
      CALL MPI_Recv(rthcuten(:,:,:), size_z, MPI_REAL, server, msg_tag(82), mpi_comm_infer_groups(1), status, ierr)
      CALL MPI_Recv(rqvcuten(:,:,:), size_z, MPI_REAL, server, msg_tag(83), mpi_comm_infer_groups(1), status, ierr)
      CALL MPI_Recv(rqccuten(:,:,:), size_z, MPI_REAL, server, msg_tag(84), mpi_comm_infer_groups(1), status, ierr)
      CALL MPI_Recv(rqrcuten(:,:,:), size_z, MPI_REAL, server, msg_tag(85), mpi_comm_infer_groups(1), status, ierr)
      CALL MPI_Recv(rqicuten(:,:,:), size_z, MPI_REAL, server, msg_tag(86), mpi_comm_infer_groups(1), status, ierr)
      CALL MPI_Recv(rqscuten(:,:,:), size_z, MPI_REAL, server, msg_tag(87), mpi_comm_infer_groups(1), status, ierr)

      write(message, *) ' my task = ', mm, ', recv from server = ', server, ', rthcuten(:2,:2,:2) =', rthcuten(:2,:2,:2)

      endif
      !call wrf_error_fatal("abor ")
      !CALL wrf_message("wrf_infer_run_cu finish")

    END SUBROUTINE  wrf_infer_run_cu  

END MODULE module_wrf_infer_cu


SUBROUTINE get_mpi_comm_infer_groups( retval, isrvr )
!<DESCRIPTION>
! This routine returns the compute+io communicator to which this
! compute task belongs for I/O server group "isrvr".
! This routine is called only by client (compute) tasks.
!</DESCRIPTION>
#if defined( DM_PARALLEL ) && !defined( STUBMPI )
      USE module_wrf_infer_cu
      IMPLICIT NONE
      INTEGER, INTENT(IN ) :: isrvr
      INTEGER, INTENT(OUT) :: retval
      retval = mpi_comm_infer_groups(isrvr)
#endif
      RETURN
END SUBROUTINE get_mpi_comm_infer_groups

SUBROUTINE wrf_infer_cu_exit
!<DESCRIPTION>
! Instruct the infer servers to shut down the WRF I/O system.
! Do not call any wrf_quilt_*() routines after this routine has been called.
! This routine is called only by client (compute) tasks.  
!</DESCRIPTION>
#if defined( DM_PARALLEL ) && ! defined (STUBMPI ) 
  USE module_wrf_infer_cu
  IMPLICIT NONE
  INCLUDE 'mpif.h'
#include "intio_tags.h"
  INTEGER                     :: actual_iserver
  INTEGER i,  tasks_in_group, comm_io_group, me, ierr 
  INTEGER iserver
  LOGICAL, EXTERNAL :: use_infer_servers

  CALL wrf_debug ( DEBUG_LVL, 'in wrf_infer_cu_exit' ) 

  IF (use_infer_servers()) THEN

    DO iserver = 1, ninfer_groups
      actual_iserver=iserver
      CALL get_mpi_comm_infer_groups( comm_io_group , actual_iserver )
      CALL mpi_x_comm_size( comm_io_group , tasks_in_group , ierr )
      CALL mpi_comm_rank( comm_io_group , me , ierr )

!   BY SENDING A NEGATIVE SIZE WE GET THE SERVERS TO SHUT DOWN
      IF ( me .eq. 0 ) then 
          CALL MPI_SSend(iserver, 1, MPI_INTEGER, tasks_in_group-1, exit_tag, comm_io_group,  ierr)
      END IF

    ENDDO
  END IF
#endif
  RETURN  
END SUBROUTINE wrf_infer_cu_exit

!<DESCRIPTION>
! Remaining routines in this file are defined outside of the module
! either to defeat arg/param type checking or to avoid an explicit use
! dependence.
!</DESCRIPTION>

SUBROUTINE disable_infering
!<DESCRIPTION>
! Call this in programs that you never want to be quilting (e.g. real)
! Must CALL before CALL to init_module_wrf_quilt().  
!</DESCRIPTION>
  USE module_wrf_infer_cu
  disable_infer = .TRUE.
  write(0,*)__FILE__,__LINE__,disable_infer, 'disable_infering'
  RETURN
END SUBROUTINE disable_infering

SUBROUTINE infering_disabled( reslt )
!<DESCRIPTION>
! Call this in programs that you never want to be quilting (e.g. real)
! Must CALL before CALL to init_module_wrf_quilt().  
!</DESCRIPTION>
  USE module_wrf_infer_cu
  LOGICAL, INTENT(OUT) :: reslt
  reslt = disable_infer
write(0,*)__FILE__,__LINE__,disable_infer
  RETURN
END SUBROUTINE infering_disabled


LOGICAL FUNCTION  use_infer_servers_for(ioform)
!<DESCRIPTION>
! Returns .TRUE. if I/O quilt servers are in-use for write operations
! AND the output servers can handle the given I/O form.  If the I/O
! form is 0, then the io form is not considered and the result is the
! same as CALLing use_output_servers.
! This routine is CALLed only by client (compute) tasks.  
!</DESCRIPTION>
  USE module_wrf_infer_cu
  integer, intent(in) :: ioform
  use_infer_servers_for = infering_enabled
  use_infer_servers_for = ( use_infer_servers_for .and. ioform<100 )
  RETURN
END FUNCTION use_infer_servers_for

LOGICAL FUNCTION  use_infer_servers()
!<DESCRIPTION>
! Returns .TRUE. if I/O quilt servers are in-use for write operations.
! This routine is CALLed only by client (compute) tasks.  
!</DESCRIPTION>
  USE module_wrf_infer_cu
  use_infer_servers = infering_enabled
  RETURN
END FUNCTION use_infer_servers

