
!----------------------------------------------------------------------------------------------
!dkay begin MSKF
!.........................................

MODULE module_cu_mskf_dl

   USE module_wrf_error
   USE module_wrf_infer_cu,only: wrf_infer_run_cu
   
!
!--------------------------------------------------------------------


CONTAINS

   SUBROUTINE inference_cu(                                  &
              ids,ide, jds,jde, kds,kde                      &
             ,ims,ime, jms,jme, kms,kme                      &
             ,its,ite, jts,jte, kts,kte                      &
             ,DT,DX,CUDT                                     &
             ,rho,RAINCV,PRATEC,NCA                          &
             ,U,V,TH,T,W,dz8w,p,pi                           &
             ,W0AVG                                          &
             ,CU_ACT_FLAG                                    &
             ,QV                                             &
             ,RTHCUTEN,RQVCUTEN,RQCCUTEN,RQRCUTEN            &
             ,RQICUTEN,RQSCUTEN                              &
             ,ZOL,HFX,UST,PBLH                               &   
             )                           
!
!-------------------------------------------------------------
   IMPLICIT NONE
!  SAVE !TWG 2017 Add to avoid memory issues
!-------------------------------------------------------------
   INTEGER,      INTENT(IN   ) ::                            &
                                  ids,ide, jds,jde, kds,kde, &
                                  ims,ime, jms,jme, kms,kme, &
                                  its,ite, jts,jte, kts,kte

   REAL,  DIMENSION( ims:ime , kms:kme , jms:jme )         , &
          INTENT(IN   ) ::                                   &
                                                          U, &
                                                          V, &
                                                          W, &
                                                         TH, &
                                                          T, &
                                                         QV, &
                                                       dz8w, &
                                                       p, &
                                                        rho, &
                                                         pi


   REAL,  DIMENSION( ims:ime , kms:kme , jms:jme )         , &
          INTENT(INOUT) ::                                   &
                                                      W0AVG   

   REAL,  INTENT(IN   ) :: DT, DX
   REAL,  INTENT(IN   ) :: CUDT

   REAL, DIMENSION( ims:ime , jms:jme ),                     &
          INTENT(INOUT) ::                           RAINCV

   REAL,    DIMENSION( ims:ime , jms:jme ),                  &
          INTENT(INOUT) ::                           PRATEC

   REAL,    DIMENSION( ims:ime , jms:jme ),                  &
            INTENT(INOUT) ::                            NCA


   LOGICAL, DIMENSION( ims:ime , jms:jme ),                  &
          INTENT(INOUT) :: CU_ACT_FLAG
!
! Optional arguments
!
   REAL, DIMENSION( ims:ime , kms:kme , jms:jme ),           &
         INTENT(INOUT) ::                                    &
                                                   RTHCUTEN, &
                                                   RQVCUTEN, &
                                                   RQCCUTEN, &
                                                   RQRCUTEN, &
                                                   RQICUTEN, &
                                                   RQSCUTEN

!ckay
   REAL, DIMENSION( ims:ime, jms:jme )                     , &
         INTENT(   IN) ::                               ZOL, &
                                                        HFX, &
                                                        UST, &
                                                       PBLH

! LOCAL VARS
!by xiaohui for running dl-based cumulus driver

    real, dimension( its:ite, jts:jte ) ::                 pratec_2d, &
                                                           nca_2d,    &
                                                           hfx_2d,    &
                                                           ust_2d,    &
                                                           pblh_2d   

    real, dimension( kms:kme, its:ite, jts:jte ) ::  w_3d

    real, dimension( kms:kme-1, its:ite, jts:jte ) ::    u_3d, &
                                                         v_3d, &
                                                         t_3d, &
                                                         qv_3d, &
                                                         p_3d, &
                                                         th_3d, &
                                                         dz8w_3d, &
                                                         rho_3d, &
                                                         pi_3d, &
                                                         w0avg_3d, &
                                                         rthcuten_3d, &
                                                         rqvcuten_3d, &
                                                         rqccuten_3d, &
                                                         rqrcuten_3d, &
                                                         rqicuten_3d, &
                                                         rqscuten_3d                                                       

    INTEGER :: i,j,k

    ! latitude loop
    do j = jts,jte
    ! longitude loop
      do i = its,ite

        pratec_2d(i,j)=pratec(i,j)
        nca_2d(i,j)=nca(i,j)
        hfx_2d(i,j)=hfx(i,j)
        ust_2d(i,j)=ust(i,j)
        pblh_2d(i,j)=pblh(i,j)

        do k = kts, kte
          w_3d(k,i,j)=w(i,k,j)
          w_3d(k+1,i,j)=w(i,k+1,j)

          u_3d(k,i,j)=u(i,k,j)
          v_3d(k,i,j)=v(i,k,j)
          t_3d(k,i,j)=t(i,k,j)
          qv_3d(k,i,j)=qv(i,k,j)
          p_3d(k,i,j)=p(i,k,j)
          th_3d(k,i,j)=th(i,k,j)
          dz8w_3d(k,i,j)=dz8w(i,k,j)
          rho_3d(k,i,j)=rho(i,k,j)
          pi_3d(k,i,j)=pi(i,k,j)
          w0avg_3d(k,i,j)=w0avg(i,k,j)

          rthcuten_3d(k,i,j)=rthcuten(i,k,j)
          rqvcuten_3d(k,i,j)=rqvcuten(i,k,j)
          rqccuten_3d(k,i,j)=rqccuten(i,k,j)
          rqrcuten_3d(k,i,j)=rqrcuten(i,k,j)
          rqicuten_3d(k,i,j)=rqicuten(i,k,j)
          rqscuten_3d(k,i,j)=rqscuten(i,k,j)

        enddo

      enddo

    enddo  


     CALL wrf_debug(0, 'Call wrf_infer_run_cu')

    CALL wrf_infer_run_cu(ite-its+1, jte-jts+1, kme-1-kms+1, kme-kms+1, &
          pratec_2d,                                                                                 &
          nca_2d,                                                                                    &
          hfx_2d,                                                                                    &   
          ust_2d,                                                                                    &
          pblh_2d,                                                                                   &   
          u_3d,                                                                                      &              
          v_3d,                                                                                      &
          w_3d,                                                                                      &
          t_3d,                                                                                      &
          qv_3d,                                                                                     &
          p_3d,                                                                                      &
          th_3d,                                                                                     &
          dz8w_3d,                                                                                   &
          rho_3d,                                                                                    &
          pi_3d,                                                                                     &
          w0avg_3d,                                                                                  &
          rthcuten_3d,                                                                               &
          rqvcuten_3d,                                                                               &
          rqccuten_3d,                                                                               &              
          rqrcuten_3d,                                                                               &
          rqicuten_3d,                                                                               &
          rqscuten_3d)                

           DO j=jts,jte
           DO i=its,ite 
               pratec(i,j)=pratec_2d(i,j)
               nca(i,j)=nca_2d(i,j)

               raincv(i,j)=dt*pratec(i,j) 

           ENDDO
           ENDDO

           DO j=jts,jte
           DO k=kts,kte
           DO i=its,ite              

              w0avg(I,K,J)=w0avg_3d(K,I,J)

              rthcuten(I,K,J)=rthcuten_3d(K,I,J)
              rqvcuten(I,K,J)=rqvcuten_3d(K,I,J)
              rqccuten(I,K,J)=rqccuten_3d(K,I,J)
              rqrcuten(I,K,J)=rqrcuten_3d(K,I,J)
              rqicuten(I,K,J)=rqicuten_3d(K,I,J)
              rqscuten(I,K,J)=rqscuten_3d(K,I,J)

           ENDDO
           ENDDO
           ENDDO

   END SUBROUTINE inference_cu


END MODULE module_cu_mskf_dl
