!------------------------------------------------------------------
MODULE module_cu_mskf_dl_replace 

  use, intrinsic :: iso_c_binding
  implicit none

    save

    interface

!--------------------------------------subroutine--------------------------

    subroutine infer_init_cu(use_gpu) bind (c)
          use, intrinsic :: iso_c_binding
          implicit none
          integer(c_int), value :: use_gpu 
    end subroutine infer_init_cu

    subroutine infer_run_cu(                                                                      &
          shape_x, shape_y, shape_z, shape_z_lev,                                                 &
          pratec,                                                                                 &
          nca,                                                                                    &
          hfx,                                                                                    &   
          ust,                                                                                    &
          pblh,                                                                                   &   
          u,                                                                                      &              
          v,                                                                                      &
          w,                                                                                      &
          t,                                                                                      &
          qv,                                                                                     &
          p,                                                                                      &
          th,                                                                                     &
          dz8w,                                                                                   &
          rho,                                                                                    &
          pi,                                                                                     &
          w0avg,                                                                                  &
          rthcuten,                                                                               &
          rqvcuten,                                                                               &
          rqccuten,                                                                               &              
          rqrcuten,                                                                               &
          rqicuten,                                                                               &
          rqscuten) bind (c)
          use, intrinsic :: iso_c_binding
          implicit none

          integer(c_int), value :: shape_x
          integer(c_int), value :: shape_y
          integer(c_int), value :: shape_z
          integer(c_int), value :: shape_z_lev
          REAL(c_float), DIMENSION( shape_x, shape_y ), INTENT(IN   )  :: pratec, &
                                                                          nca,    &
                                                                          hfx,    &
                                                                          ust,    &
                                                                          pblh    

          REAL(c_float), DIMENSION(shape_z_lev, shape_x, shape_y), INTENT(IN) :: w

          REAL(c_float), DIMENSION(shape_z, shape_x, shape_y), INTENT(IN) :: u, &
                                                                             v, &
                                                                             t, &
                                                                             qv, &
                                                                             p, &
                                                                             th, &
                                                                             dz8w, &
                                                                             rho, &
                                                                             pi, &
                                                                             w0avg, &
                                                                             rthcuten, &
                                                                             rqvcuten, &
                                                                             rqccuten, &
                                                                             rqrcuten, &
                                                                             rqicuten, &
                                                                             rqscuten

    end subroutine infer_run_cu

    subroutine save_fortran_array2(data_array, in_x, in_y, filename) bind (c)
        use, intrinsic :: iso_c_binding
        implicit none
        integer(c_int), value :: in_x
        integer(c_int), value :: in_y
        REAL(c_float), DIMENSION( in_x, in_y ), INTENT(IN)  :: data_array
        CHARACTER(c_char), DIMENSION(*) :: filename
    end subroutine save_fortran_array2

    subroutine save_fortran_array3(data_array, in_x, in_y, in_z, filename) bind (c)
        use, intrinsic :: iso_c_binding
        implicit none
        integer(c_int), value :: in_x
        integer(c_int), value :: in_y
        integer(c_int), value :: in_z
        REAL(c_float), DIMENSION( in_x, in_y, in_z), INTENT(IN)  :: data_array
        CHARACTER(c_char), DIMENSION(*) :: filename
    end subroutine save_fortran_array3

    subroutine save_fortran_array(                                                                &
          ims, ime, jms, jme, kms, kme,                                                           &
          its, ite, jts, jte, kts, kte,                                                           &
          pratec,                                                                                 &
          nca,                                                                                    &
          hfx,                                                                                    &   
          ust,                                                                                    &
          pblh,                                                                                   &   
          u,                                                                                      &              
          v,                                                                                      &
          w,                                                                                      &
          t,                                                                                      &
          qv,                                                                                     &
          p,                                                                                      &
          th,                                                                                     &
          dz8w,                                                                                   &
          rho,                                                                                    &
          pi,                                                                                     &
          w0avg,                                                                                  &
          rthcuten,                                                                               &
          rqvcuten,                                                                               &
          rqccuten,                                                                               &              
          rqrcuten,                                                                               &
          rqicuten,                                                                               &
          rqscuten,                                                                               &
          nca_out,                                                                                &
          pratec_out,                                                                             &
          rthcuten_out,                                                                           &
          rqvcuten_out,                                                                           &
          rqccuten_out,                                                                           &              
          rqrcuten_out,                                                                           &
          rqicuten_out,                                                                           &
          rqscuten_out,                                                                           &
          filename) bind (c)
          use, intrinsic :: iso_c_binding
          implicit none

          integer(c_int), value :: ims
          integer(c_int), value :: ime
          integer(c_int), value :: jms
          integer(c_int), value :: jme
          integer(c_int), value :: kms
          integer(c_int), value :: kme

          integer(c_int), value :: its
          integer(c_int), value :: ite
          integer(c_int), value :: jts
          integer(c_int), value :: jte
          integer(c_int), value :: kts
          integer(c_int), value :: kte

          REAL(c_float), DIMENSION( ims:ime, jms:jme ), INTENT(IN   )  :: pratec, &
                                                                          nca,    &
                                                                          hfx,    &
                                                                          ust,    &
                                                                          pblh,   &
                                                                          nca_out,&
                                                                          pratec_out                               

          REAL(c_float), DIMENSION(kms:kme-1, ims:ime, jms:jme), INTENT(IN) :: u, &
                                                                               v, &
                                                                               t, &
                                                                               qv, &
                                                                               p, &
                                                                               th, &
                                                                               dz8w, &
                                                                               rho, &
                                                                               pi, &
                                                                               w0avg, &
                                                                               rthcuten, &
                                                                               rqvcuten, &
                                                                               rqccuten, &
                                                                               rqrcuten, &
                                                                               rqicuten, &
                                                                               rqscuten, &
                                                                               rthcuten_out, &                                             
                                                                               rqvcuten_out, &
                                                                               rqccuten_out, &
                                                                               rqrcuten_out, &
                                                                               rqicuten_out, &
                                                                               rqscuten_out         

          REAL(c_float), DIMENSION(kms:kme, ims:ime, jms:jme), INTENT(IN) :: w
                                                                            
          CHARACTER(c_char), DIMENSION(*) :: filename

    end subroutine save_fortran_array

  end interface

  CONTAINS
    subroutine fillOneDimArray_atfortran (a)      
        implicit none      
        real,dimension (:), intent (out) :: a      

        ! local variables     
        integer :: i, arraySize  
        arraySize = size(a)
        do i = 1, arraySize         
            a(i) = 2.0      
        end do  

    end subroutine fillOneDimArray_atfortran 

    subroutine printOneDimArray_atfortran(a)
        implicit none

        real,dimension (:) :: a  
        integer::i, arraySize
        !arraySize = size(a)
        !print *, "size=", arraySize, ":", a
        arraySize = 10
        do i = 1, arraySize
            Print *, a(i)
            !WRITE(wrf_model_infer_err_message, *),a(i) 
            !call wrf_debug(0,wrf_model_infer_err_message)
        end do

    end subroutine printOneDimArray_atfortran

END MODULE module_cu_mskf_dl_replace
