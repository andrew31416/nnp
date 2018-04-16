module feature_util
    use config
    use feature_config

    implicit none

    !* blas/lapack
    external :: dcopy
    external :: dgemm
    external :: dgetrf
    external :: dgetri
    real(8),external :: ddot

    contains

        subroutine invert_matrix(mtrx_in,mtrx_out)
            implicit none
            
            real(8),intent(in) :: mtrx_in(1:3,1:3)
            real(8),intent(inout) :: mtrx_out(1:3,1:3)

            !real(8) :: det
            
            integer :: ipiv(1:4),info
            real(8) :: lwork(1:3,1:3)
            !real(8) :: tmp(1:3,1:3)            
            
            !mtrx_out(1,1) = mtrx_in(2,2)*mtrx_in(3,3) - mtrx_in(3,2)*mtrx_in(2,3)
            !mtrx_out(2,1) = mtrx_in(2,3)*mtrx_in(3,1) - mtrx_in(2,1)*mtrx_in(3,3)
            !mtrx_out(3,1) = mtrx_in(2,1)*mtrx_in(3,2) - mtrx_in(3,1)*mtrx_in(2,2) 

            !mtrx_out(1,2) = mtrx_in(3,2)*mtrx_in(1,3) - mtrx_in(1,2)*mtrx_in(3,3)
            !mtrx_out(2,2) = mtrx_in(1,1)*mtrx_in(3,3) - mtrx_in(3,1)*mtrx_in(1,3)
            !mtrx_out(3,2) = mtrx_in(3,1)*mtrx_in(1,2) - mtrx_in(1,1)*mtrx_in(3,2)

            !mtrx_out(1,3) = mtrx_in(1,2)*mtrx_in(2,3) - mtrx_in(2,2)*mtrx_in(1,3)
            !mtrx_out(2,3) = mtrx_in(2,1)*mtrx_in(1,3) - mtrx_in(1,1)*mtrx_in(2,3)
            !mtrx_out(3,3) = mtrx_in(1,1)*mtrx_in(2,2) - mtrx_in(2,1)*mtrx_in(1,2)

            !det = mtrx_in(1,1)*mtrx_out(1,1) + mtrx_in(1,2)*mtrx_out(2,1) + &
            !        &mtrx_in(1,3)*mtrx_out(3,1)

            !mtrx_out(:,:) = mtrx_out(:,:)/det                  
            mtrx_out(:,:) = mtrx_in(:,:)
            call dgetrf(3,3,mtrx_out,3,ipiv,info)
            call dgetri(3,mtrx_out,3,ipiv,lwork,3,info)
        end subroutine

        real(8) function matrix_determinant(mtrx)
            implicit none

            real(8),intent(in) :: mtrx(:,:)
            integer :: dim(1:2),n,ii
            real(8) :: det,tmp(1:3)

            dim = shape(mtrx)

            !* dimension 
            n = dim(1)
            det = 0
            
            if (n.eq.1) then
               det = mtrx(1,1)
            else if (n.eq.2) then
               det = mtrx(1,1)*mtrx(2,2) - mtrx(1,2)*mtrx(2,1)
            else if (n.eq.3) then
                tmp(1) = mtrx(2,2)*mtrx(3,3) - mtrx(3,2)*mtrx(2,3)
                tmp(2) = mtrx(2,3)*mtrx(3,1) - mtrx(2,1)*mtrx(3,3)
                tmp(3) = mtrx(2,1)*mtrx(3,2) - mtrx(3,1)*mtrx(2,2)

                det = 0.0d0
                do ii=1,3,1
                    det = det + mtrx(1,ii)*tmp(ii)
                end do
            else
                call error("matrix_determinant","unsupported dimension")    
            end if

            matrix_determinant = det

        end function matrix_determinant

        subroutine matrix_vec_mult(cell,frac,cart)
            implicit none

            real(8),intent(in) :: frac(1:3),cell(1:3,1:3)
            real(8),intent(out) :: cart(1:3)

            !* scratch
            integer :: ii,jj

            do ii=1,3
                cart(ii) = 0.0d0
                do jj=1,3
                    cart(ii) = cart(ii) + cell(ii,jj)*frac(jj)
                end do
            end do
        end subroutine matrix_vec_mult

        subroutine project_image(frac,cell,ii,jj,kk,cart)
            implicit none

            real(8),intent(in) :: frac(1:3),cell(1:3,1:3)
            integer,intent(in) :: ii,jj,kk
            real(8),intent(out) :: cart(1:3)

            !* scratch
            real(8) :: tmp(1:3)

            tmp(:) = frac(:)

            tmp(1) = tmp(1) + dble(ii)
            tmp(2) = tmp(2) + dble(jj)
            tmp(3) = tmp(3) + dble(kk)

            call matrix_vec_mult(cell,tmp,cart)
        end subroutine project_image

        subroutine get_ultracell(rcut,maxsize,set_type,conf,ultracart,ultraidx,ultraZ)
            !============================================================!
            ! return ultracell of cartesian positions from local         !
            ! cartesian positions                                        !
            !
            ! localcell(i,j) = ith cartesian coordinate of jth vector    !
            !============================================================!

            implicit none

            integer,intent(in) :: maxsize,set_type,conf
            real(8),intent(in) :: rcut
            real(8),allocatable,intent(inout) :: ultracart(:,:),ultraZ(:)
            integer,allocatable,intent(inout) :: ultraidx(:)

            !* scratch
            integer :: layer,maxiter,atomii,atomjj,ii,jj,kk,cntr
            integer :: ultranatm,tmpidx(1:maxsize)
            real(8) :: fracii(1:3),cartii(1:3),cartjj(1:3),rcut2
            real(8) :: tmpcart(1:3,1:maxsize),tmpZ(1:maxsize)
            logical :: pairfound
            real(8) :: localcell(1:3,1:3),invcell(1:3,1:3)
            integer :: natm

            
            ultranatm = 0

            rcut2 = rcut**2
            maxiter = 10

            !* local cell
            localcell(:,:) = data_sets(set_type)%configs(conf)%cell(:,:)

            !* inverse of local cell
            call invert_matrix(localcell,invcell)

            !* atoms in cell
            natm = data_sets(set_type)%configs(conf)%n

            do layer=1,maxiter
                cntr = 0

                do atomii=1,natm
                    !* cartesians of central atom
                    cartii(1:3) = data_sets(set_type)%configs(conf)%r(1:3,atomii)

                    !* fractional coordinates fracii
                    call matrix_vec_mult(invcell,cartii,fracii)


                    !* search over all layers
                    do ii=-layer,layer
                        do jj=-layer,layer
                            do kk=-layer,layer
                
                                !* only if one atom pair with local cell is
                                !* found do we add this image of atomii
                                pairfound = .false.

                                do atomjj=1,natm
                                    if ( (atomjj.eq.atomii).and.(ii.eq.0).and.(jj.eq.0).and.(kk.eq.0) ) then
                                        !* don't want to include interactions with oneself
                                        cycle
                                    end if

                                    !* cartesians of atomjj
                                    cartjj(:) = data_sets(set_type)%configs(conf)%r(:,atomjj)
                                    
                                    if ( (abs(ii).ne.layer).and.(abs(jj).ne.layer).and.(abs(kk).ne.layer) ) then
                                        !* have already considered this layer of atoms
                                        cycle
                                    end if

                                    !* get cartesian of projected atomii
                                    call project_image(fracii,localcell,ii,jj,kk,cartii)
                                 
                                    if ( (cartii(1)-cartjj(1))**2+(cartii(2)-cartjj(2))**2+&
                                            &(cartii(3)-cartjj(3))**2 .le. rcut2 ) then
                                        pairfound = .true.
                                    end if
                                end do
                                        
                                if (pairfound) then
                                    cntr = cntr + 1
                                    ultranatm = ultranatm + 1
                                    
                                    if (ultranatm.gt.maxsize) then
                                        write(*,*) 'Warning - more than',maxsize,'ultracell atoms found',&
                                                &'. Increase maxsize to continue'
                                        call exit(0)
                                    end if

                                    !* store cartesian coordinates
                                    tmpcart(1:3,ultranatm) = cartii(1:3)

                                    !* store idx of local atom found
                                    tmpidx(ultranatm) = atomii

                                    !* store atomic number 
                                    tmpZ(ultranatm) = data_sets(set_type)%configs(conf)%z(atomii)
                                end if ! pair found 
                            end do ! end kk
                        end do ! end jj

                    end do ! end ii
                end do ! end atomii

                if (cntr.eq.0) then
                    !* no further neighbours will be found
                    exit
                end if
            end do ! end layer
            
            !* now add atoms in local cell
            do atomii=1,natm
                ultranatm = ultranatm + 1
                
                !* store cartesian coordinates
                tmpcart(1:3,ultranatm) = data_sets(set_type)%configs(conf)%r(1:3,atomii)

                !* store corresponding local cell idx
                tmpidx(ultranatm) = atomii

                !* store atomic number 
                tmpZ(ultranatm) = data_sets(set_type)%configs(conf)%z(atomii)
            end do
            
            allocate (ultracart(3,ultranatm))
            allocate (ultraidx(ultranatm))
            allocate (ultraZ(ultranatm))
            ultracart(1:3,1:ultranatm) = tmpcart(1:3,1:ultranatm)
            ultraidx(1:ultranatm) = tmpidx(1:ultranatm)
            ultraZ(1:ultranatm) = tmpZ(1:ultranatm)


        end subroutine get_ultracell

        real(8) function maxrcut(arg)
            !===============================================================!
            ! Return maximum cut off radius of all current features         !
            !                                                               !
            ! Input                                                         !
            ! -----                                                         !
            !   - arg : 0 = max of all features                             !
            !           1 = max of all isotropic features                   !
            !           2 = max of all anisotropic features                 !
            !===============================================================!
            
            implicit none

            integer,intent(in) :: arg

            real(8),allocatable :: tmprcut(:)
            integer :: ii,ftype
            real(8) :: tmpr

            allocate(tmprcut(feature_params%num_features))

            do ii=1,feature_params%num_features,1
                !* feature type
                ftype = feature_params%info(ii)%ftype

                !* interaction cut off (can be null)
                tmpr = feature_params%info(ii)%rcut

                if (arg.eq.0) then
                    if (ftype.eq.featureID_StringToInt("atomic_number")) then
                        tmprcut(ii) = -1.0d0
                    else
                        !* all features
                        tmprcut(ii) = tmpr
                    end if
                else if (arg.eq.1) then
                    if (feature_IsTwoBody(ftype)) then
                        tmprcut(ii) = tmpr
                    else
                        tmprcut(ii) = -1.0d0
                    end if
                else if (arg.eq.2) then
                    if ( (feature_IsTwoBody(ftype).neqv..true.).and.(ftype.ne.&
                    &featureID_StringToInt("atomic_number")) ) then
                        tmprcut(ii) = tmpr
                    else
                        tmprcut(ii) = -1.0d0
                    end if
                end if
            end do

            tmpr = maxval(tmprcut)
            deallocate(tmprcut)
    
            maxrcut = tmpr
        end function maxrcut

        logical function threebody_features_present()
            !===============================================================!
            ! check if any three body features are present                  ! 
            !===============================================================!
            
            implicit none

            logical :: found_some
            integer :: ii,ftype,tmp(1:3)

            found_some = .false.

            tmp(1) = featureID_StringToInt("acsf_behler-g4")
            tmp(2) = featureID_StringToInt("acsf_behler-g5")
            tmp(3) = featureID_StringToInt("acsf_normal-b3")

            do ii=1,feature_params%num_features,1
                ftype = feature_params%info(ii)%ftype

                if ( (ftype.eq.tmp(1)).or.(ftype.eq.tmp(2)).or.(ftype.eq.tmp(3)) ) then
                    found_some = .true.
                    exit
                end if
            end do

            threebody_features_present = found_some
        end function threebody_features_present

        subroutine calculate_twobody_info(set_type,conf,ultracart,ultraz,ultraidx)
            !===============================================================!
            !* calculate isotropic atom-atom distances and derivatives     *!
            !===============================================================!
            use tapering, only : taper_1,taper_deriv_1

            implicit none

            real(8),intent(in) :: ultracart(:,:),ultraz(:)
            integer,intent(in) :: ultraidx(:),set_type,conf

            !* scratch
            integer :: dim(1:1),ii,jj,cntr
            real(8) :: rcut2,dr2,rtol2,rcut
            real(8) :: drii(1:3),drjj(1:3)
            real(8) :: fs=0.d0
        
            if (speedup_applies("twobody_rcut")) then
                fs = get_Nbody_common_fs(2)
            end if
            if (speedup_applies("keep_all_neigh_info")) then
                if (allocated(set_neigh_info(conf)%twobody)) then
                    call error("calculate_twobody_info","have needlessly called this function")
                end if
            end if

            !* neighbour info for all atoms in conf
            allocate(set_neigh_info(conf)%twobody(data_sets(set_type)%configs(conf)%n))

            !* dim(1) = number atoms in ultra cell
            dim = shape(ultraidx)

            !* max isotropic interaction cut off
            rcut = maxrcut(1)
            rcut2 = rcut**2

            !* min distance between 2 different atoms allowed
            rtol2 = (0.0000001)**2

            !* info for each atom
            !allocate(feature_isotropic(data_sets(set_type)%configs(conf)%n))

            do ii=1,data_sets(set_type)%configs(conf)%n,1
                !* iterate over local atoms

                !* local position
                drii(:) = data_sets(set_type)%configs(conf)%r(:,ii)

                cntr = 0
                do jj=1,dim(1),1
                    drjj(:) = ultracart(:,jj)

                    dr2 = distance2(drii,drjj) 

                    if ( (dr2.lt.rtol2).or.(dr2.gt.rcut2) ) then
                        !* same atom or beyond cut off
                        cycle
                    else 
                        cntr = cntr + 1    
                    end if
                end do


                ! redundant v
                !* allocate neighbour mem
                !allocate(feature_isotropic(ii)%dr(cntr))
                !allocate(feature_isotropic(ii)%idx(cntr))
                !allocate(feature_isotropic(ii)%z(cntr))
                !allocate(feature_isotropic(ii)%drdri(3,cntr))
                !if (speedup_applies("twobody_rcut")) then
                !    !* all two body feature share same rcut,fs
                !    allocate(feature_isotropic(ii)%dr_taper(cntr))
                !    allocate(feature_isotropic(ii)%dr_taper_deriv(cntr))
                !end if
                !!* number of neighbours 
                !feature_isotropic(ii)%n = cntr
                !
                !!* atomic number of central atom in interaction
                !feature_isotropic(ii)%z_atom = data_sets(set_type)%configs(conf)%z(ii)
                ! redundant ^

                allocate(set_neigh_info(conf)%twobody(ii)%dr(cntr))
                allocate(set_neigh_info(conf)%twobody(ii)%idx(cntr))
                allocate(set_neigh_info(conf)%twobody(ii)%z(cntr))
                allocate(set_neigh_info(conf)%twobody(ii)%drdri(3,cntr))
                if (speedup_applies("twobody_rcut")) then
                    !* all two body feature share same rcut,fs
                    allocate(set_neigh_info(conf)%twobody(ii)%dr_taper(cntr))
                    allocate(set_neigh_info(conf)%twobody(ii)%dr_taper_deriv(cntr))
                end if
                set_neigh_info(conf)%twobody(ii)%n = cntr
                set_neigh_info(conf)%twobody(ii)%z_atom = data_sets(set_type)%configs(conf)%z(ii)
               
                cntr = 1
                do jj=1,dim(1),1
                    drjj(:) = ultracart(:,jj)

                    dr2 = distance2(drii,drjj) 

                    if ( (dr2.lt.rtol2).or.(dr2.gt.rcut2) ) then
                        !* same atom or beyond cut off
                        cycle
                    else 
                        !* atom-atom distance
                        !feature_isotropic(ii)%dr(cntr) = sqrt(dr2) ! DEPRECATED
                        set_neigh_info(conf)%twobody(ii)%dr(cntr) = sqrt(dr2)

                        if (speedup_applies("twobody_rcut")) then
                            !feature_isotropic(ii)%dr_taper(cntr) = taper_1(&             ! DEP. 
                            !        &feature_isotropic(ii)%dr(cntr),rcut,fs)             ! DEP.
                            !feature_isotropic(ii)%dr_taper_deriv(cntr) = taper_deriv_1(& ! DEP.
                            !        &feature_isotropic(ii)%dr(cntr),rcut,fs)             ! DEP.
                            
                            set_neigh_info(conf)%twobody(ii)%dr_taper(cntr) = taper_1(&
                                    &set_neigh_info(conf)%twobody(ii)%dr(cntr),rcut,fs)
                            set_neigh_info(conf)%twobody(ii)%dr_taper_deriv(cntr) = taper_deriv_1(&
                                    &set_neigh_info(conf)%twobody(ii)%dr(cntr),rcut,fs)
                        end if

                        !* local cell identifier of neighbour cntr
                        !feature_isotropic(ii)%idx(cntr) = ultraidx(jj) ! DEP.
                        set_neigh_info(conf)%twobody(ii)%idx(cntr) = ultraidx(jj)

                        !* d rij / drj 
                        !feature_isotropic(ii)%drdri(:,cntr) = (drjj(:) - drii(:)) / & ! DEP.
                        !        &feature_isotropic(ii)%dr(cntr)                       ! DEP.
                        
                        set_neigh_info(conf)%twobody(ii)%drdri(:,cntr) = (drjj(:) - drii(:)) / &
                                &set_neigh_info(conf)%twobody(ii)%dr(cntr)
                        
                        !* Z of neighbour cntr
                        !feature_isotropic(ii)%z(cntr) = ultraz(jj) ! DEP. 
                        set_neigh_info(conf)%twobody(ii)%z(cntr) = ultraz(jj) 
                        
                        cntr = cntr + 1    
                    end if
                end do


            end do
        end subroutine calculate_twobody_info
        
       
        logical function feat_doesnt_taper_drjk(ft_idx)
            integer,intent(in) :: ft_idx
            
            logical :: res

            res = .false.

            !if (feature_params%info(ft_idx)%ftype.eq.featureID_StringToInt("acsf_behler-g5").or.&
            !&(feature_params%info(ft_idx)%ftype.eq.featureID_StringToInt("acsf_normal-b3")) ) then
            if (feature_params%info(ft_idx)%ftype.eq.featureID_StringToInt("acsf_behler-g5")) then
                res = .true.
            end if
            feat_doesnt_taper_drjk = res
        end function feat_doesnt_taper_drjk
        
        subroutine calculate_threebody_info(set_type,conf,ultracart,ultraz,ultraidx)
            !===============================================================!
            !* calculate isotropic atom-atom distances and derivatives     *!
            !===============================================================!
            use tapering, only : taper_1,taper_deriv_1

            implicit none

            real(8),intent(in) :: ultracart(:,:),ultraz(:)
            integer,intent(in) :: ultraidx(:),set_type,conf

            !* scratch
            integer :: dim(1:1),ii,jj,kk,zz,cntr
            real(8) :: rcut2,dr2ij,dr2ik,dr2jk,rtol2
            real(8) :: rii(1:3),rjj(1:3),rkk(1:3),drij,drik,drjk
            real(8) :: drij_vec(1:3),drik_vec(1:3),drjk_vec(1:3)
            real(8) :: sign_ij(1:3),sign_ik(1:3),rcut,fs
            integer :: maxbuffer
            type(feature_info_threebody) :: aniso_info
            logical :: any_rjk
            
            if (.not.allocated(set_neigh_info)) then
                call error("calculate_threebody_info","have failed to allcoate set_neigh_info")
            else
                if (allocated(set_neigh_info(conf)%threebody)) then
                    if (speedup_applies("keep_all_neigh_info")) then
                        call error("calculate_threebody_info","have needlessly called this func.")
                    else
                        call error("calculate_threebody_info",&
                                &"have failed to deallocate 3body info")
                    end if
                end if
            end if

            !* neighbour info for all atoms in conf
            allocate(set_neigh_info(conf)%threebody(data_sets(set_type)%configs(conf)%n))
            
            !* dim(1) = number atoms in ultra cell
            dim = shape(ultraidx)

            !* max anisotropic interaction cut off
            rcut = maxrcut(2)
            rcut2 = rcut**2
            
            !* min distance between 2 different atoms allowed
            rtol2 = (0.0000001)**2

            !* max number of assumed 3-body terms per atom
            maxbuffer = 50000
            
            if (speedup_applies("threebody_rcut")) then
                fs = get_Nbody_common_fs(3)
            end if


            !* structure for all three body info associated with structure
            !allocate(feature_threebody_info(data_sets(set_type)%configs(conf)%n))
            
            allocate(aniso_info%cos_ang(maxbuffer))
            allocate(aniso_info%dr(3,maxbuffer))
            allocate(aniso_info%z(2,maxbuffer))
            allocate(aniso_info%idx(2,maxbuffer))
            allocate(aniso_info%dcos_dr(3,3,maxbuffer))
            allocate(aniso_info%drdri(3,6,maxbuffer))
            if (speedup_applies("threebody_rcut")) then
                allocate(aniso_info%dr_taper(3,maxbuffer))
                allocate(aniso_info%dr_taper_deriv(3,maxbuffer))
            end if

            any_rjk = .false.
            do ii=1,D
                if (feat_doesnt_taper_drjk(ii)) then
                    !* behler g-5 and normal b-3 have no constraint (tapering) on drjk
                    any_rjk = .true.
                end if
            end do


            do ii=1,data_sets(set_type)%configs(conf)%n,1
                !* iterate over local atoms

                !* local position
                rii(:) = data_sets(set_type)%configs(conf)%r(:,ii)

                cntr = 0
                do jj=1,dim(1),1
                    rjj(:) = ultracart(:,jj)

                    dr2ij = distance2(rii,rjj) 

                    if ( (dr2ij.lt.rtol2).or.(dr2ij.gt.rcut2) ) then
                        !* same atom or beyond cut off
                        cycle
                    end if

                    drij = sqrt(dr2ij)
                    drij_vec = rjj - rii
        

                    !* look for unique three-body terms
                    do kk=jj+1,dim(1),1
                        if (jj.eq.kk) then
                            !* same atom
                            cycle
                        end if
                        rkk(:) = ultracart(:,kk)

                        dr2ik = distance2(rii,rkk) 
                        dr2jk = distance2(rjj,rkk)
                        
                        if ( (dr2ik.lt.rtol2).or.(dr2ik.gt.rcut2).or.(dr2jk.lt.rtol2) ) then
                            cycle
                        else if ( (any_rjk.neqv..true.).and.(dr2jk.gt.rcut2) ) then
                            !* only behler g-5 and normal b-3 do not taper drjk
                            cycle
                        end if

                        drik = sqrt(dr2ik)
                        drjk = sqrt(dr2jk)
                        drik_vec = rkk - rii
                        drjk_vec = rkk - rjj

                        !* have found a three-body term
                        cntr = cntr + 1  
                        if (cntr.gt.maxbuffer) then
                            call error("calculate_threebody_info",&
                                    &"value of maxbuffer too small, increase size")
                        end if
                            
                       
                        !* cos(dtheta_{ijk})
                        aniso_info%cos_ang(cntr) = cos_angle(rjj-rii,rkk-rii,drij,drik)

                        !* displacements
                        aniso_info%dr(1,cntr) = drij    ! central vs. jj
                        aniso_info%dr(2,cntr) = drik    ! central vs. kk
                        aniso_info%dr(3,cntr) = drjk    ! jj vs. kk
          
                        !* tapering
                        if (speedup_applies("threebody_rcut")) then
                            aniso_info%dr_taper(1,cntr) = taper_1(drij,rcut,fs)
                            aniso_info%dr_taper(2,cntr) = taper_1(drik,rcut,fs)
                            aniso_info%dr_taper(3,cntr) = taper_1(drjk,rcut,fs)
                            aniso_info%dr_taper_deriv(1,cntr) = taper_deriv_1(drij,rcut,fs)
                            aniso_info%dr_taper_deriv(2,cntr) = taper_deriv_1(drik,rcut,fs)
                            aniso_info%dr_taper_deriv(3,cntr) = taper_deriv_1(drjk,rcut,fs)
                        end if 
           
                        !* atomic number
                        aniso_info%z(1,cntr) = ultraz(jj)
                        aniso_info%z(2,cntr) = ultraz(kk)
                   
                        !* local cell identifier
                        aniso_info%idx(1,cntr) = ultraidx(jj)
                        aniso_info%idx(2,cntr) = ultraidx(kk)
                  
                        !---------------------------------!
                        !* atom-atom distance derivative *!
                        !---------------------------------!
                 
                        ! d |rj-ri| / drj
                        if (ii.ne.ultraidx(jj)) then
                            ! ii != jj

                            if (ultraidx(jj).eq.ultraidx(kk)) then
                                ! ii!=jj , jj==kk, ii!=kk
                                aniso_info%drdri(:,1,cntr) = drij_vec / drij  ! d |rj-ri| / drj
                                aniso_info%drdri(:,2,cntr) = drij_vec / drij  ! d |rj-ri| / drk
                                aniso_info%drdri(:,3,cntr) = drik_vec / drik  ! d |rk-ri| / drk
                                aniso_info%drdri(:,4,cntr) = drik_vec / drik  ! d |rk-ri| / drj
                                aniso_info%drdri(:,5,cntr) = 0.0d0            ! d |rk-rj| / drk 
                                aniso_info%drdri(:,6,cntr) = 0.0d0            ! d |rk-rj| / dri
                                
                                sign_ij(1) =  1.0d0 
                                sign_ij(2) =  1.0d0
                                sign_ij(3) = -1.0d0
                                
                                sign_ik(1) =  1.0d0
                                sign_ik(2) =  1.0d0
                                sign_ik(3) = -1.0d0
                            else if (ii.eq.ultraidx(kk)) then
                                ! ii!=jj , ii==kk, jj!=kk
                                aniso_info%drdri(:,1,cntr) = drij_vec / drij  ! d |rj-ri| / drj
                                aniso_info%drdri(:,2,cntr) = -drij_vec / drij ! d |rj-ri| / drk
                                aniso_info%drdri(:,3,cntr) = 0.d0             ! d |rk-ri| / drk
                                aniso_info%drdri(:,4,cntr) = 0.0d0            ! d |rk-ri| / drj
                                aniso_info%drdri(:,5,cntr) = drjk_vec / drjk  ! d |rk-rj| / drk
                                aniso_info%drdri(:,6,cntr) = drjk_vec / drjk  ! d |rk-rj| / dri
                                
                                sign_ij(1) =  1.0d0
                                sign_ij(2) = -1.0d0
                                sign_ij(3) = -1.0d0

                                sign_ik(:) =  0.0d0
                            else if ( (ii.ne.ultraidx(kk)).and.(ultraidx(jj).ne.ultraidx(kk)) ) then
                                ! ii!=jj, jj!=kk, ii!=kk 
                                aniso_info%drdri(:,1,cntr) = drij_vec / drij  ! d |rj-ri| / drj
                                aniso_info%drdri(:,2,cntr) = 0.0d0            ! d |rj-ri| / drk
                                aniso_info%drdri(:,3,cntr) = drik_vec / drik  ! d |rk-ri| / drk
                                aniso_info%drdri(:,4,cntr) = 0.0d0            ! d |rk-ri| / drj
                                aniso_info%drdri(:,5,cntr) = drjk_vec / drjk  ! d |rk-rj| / drk
                                aniso_info%drdri(:,6,cntr) = 0.0d0            ! d |rk-rj| / dri
                                
                                sign_ij(1) =  1.0d0
                                sign_ij(2) =  0.0d0
                                sign_ij(3) = -1.0d0

                                sign_ik(1) =  0.0d0
                                sign_ik(2) =  1.0d0
                                sign_ik(3) = -1.0d0
                            end if
                        else
                            if (ii.eq.ultraidx(kk)) then
                                ! ii==jj==kk
                                aniso_info%drdri(:,1,cntr) = 0.0d0            ! d |rj-ri| / drj
                                aniso_info%drdri(:,2,cntr) = 0.0d0            ! d |rj-ri| / drk
                                aniso_info%drdri(:,3,cntr) = 0.0d0            ! d |rk-ri| / drk
                                aniso_info%drdri(:,4,cntr) = 0.0d0            ! d |rk-ri| / drj
                                aniso_info%drdri(:,5,cntr) = 0.0d0            ! d |rk-rj| / drk
                                aniso_info%drdri(:,6,cntr) = 0.0d0            ! d |rk-rj| / dri
                                
                                sign_ij(:) = 0.0d0
                                sign_ik(:) = 0.0d0
                            else if ( (ultraidx(jj).ne.ultraidx(kk)).and.(ii.ne.ultraidx(kk)) ) then
                                ! ii==jj , ii!=kk, jj!=kk
                                aniso_info%drdri(:,1,cntr) = 0.0d0                ! d |rj-ri| / drj
                                aniso_info%drdri(:,2,cntr) = 0.0d0                ! d |rj-ri| / drk
                                aniso_info%drdri(:,3,cntr) = drik_vec / drik      ! d |rk-ri| / drk
                                aniso_info%drdri(:,4,cntr) = -drik_vec / drik     ! d |rk-ri| / drj
                                aniso_info%drdri(:,5,cntr) = drjk_vec / drjk      ! d |rk-rj| / drk
                                aniso_info%drdri(:,6,cntr) = -drjk_vec / drjk     ! d |rk-rj| / dri
                                
                                sign_ij(:) =  0.0d0
                                sign_ik(1) =  -1.0d0
                                sign_ik(2) =  1.0d0
                                sign_ik(3) = -1.0d0
                            end if
                        end if
    !aniso_info%drdri(:,1,cntr) = drij_vec / drij  ! d |rj-ri| / drj
    !aniso_info%drdri(:,2,cntr) = 0.0d0            ! d |rj-ri| / drk
    !aniso_info%drdri(:,3,cntr) = drik_vec / drik  ! d |rk-ri| / drk
    !aniso_info%drdri(:,4,cntr) = 0.0d0            ! d |rk-ri| / drj
    !aniso_info%drdri(:,5,cntr) = drjk_vec / drjk  ! d |rk-rj| / drk
    !aniso_info%drdri(:,6,cntr) = 0.0d0            ! d |rk-rj| / dri
    !
    !sign_ij(1) =  1.0d0
    !sign_ij(2) =  0.0d0
    !sign_ij(3) = -1.0d0

    !sign_ik(1) =  0.0d0
    !sign_ik(2) =  1.0d0
    !sign_ik(3) = -1.0d0

                        ! remember that d |rj-ri| / dri = - d |rj-ri| / drj
                   
                        !---------------------!
                        !* cosine derivative *!
                        !---------------------!
                      
                        ! j=1 , k=2 , i=3
                        do zz=1,3,1
                            aniso_info%dcos_dr(:,zz,cntr) = 1.0d0/(drij*drik) * ( drij_vec*&
                                    &(sign_ik(zz)-aniso_info%cos_ang(cntr)*sign_ij(zz)*drik/drij) + &
                                    &drik_vec*(sign_ij(zz) - aniso_info%cos_ang(cntr)*&
                                    &sign_ik(zz)*drij/drik) )
                        end do
                        
                    end do !* end loop kk over second neighbours
                end do !* end loop jj over first neighbours
                

                !* now copy into DEP. vv
                !allocate(feature_threebody_info(ii)%cos_ang(cntr))
                !allocate(feature_threebody_info(ii)%dr(3,cntr))
                !allocate(feature_threebody_info(ii)%z(2,cntr))
                !allocate(feature_threebody_info(ii)%idx(2,cntr))
                !allocate(feature_threebody_info(ii)%dcos_dr(3,3,cntr))
                !allocate(feature_threebody_info(ii)%drdri(3,6,cntr))
                !if (speedup_applies("threebody_rcut")) then
                !    allocate(feature_threebody_info(ii)%dr_taper(3,cntr))
                !    allocate(feature_threebody_info(ii)%dr_taper_deriv(3,cntr))
                !end if
                !! DEP ^^
                
                allocate(set_neigh_info(conf)%threebody(ii)%cos_ang(cntr))
                allocate(set_neigh_info(conf)%threebody(ii)%dr(3,cntr))
                allocate(set_neigh_info(conf)%threebody(ii)%z(2,cntr))
                allocate(set_neigh_info(conf)%threebody(ii)%idx(2,cntr))
                allocate(set_neigh_info(conf)%threebody(ii)%dcos_dr(3,3,cntr))
                allocate(set_neigh_info(conf)%threebody(ii)%drdri(3,6,cntr))
                if (speedup_applies("threebody_rcut")) then
                    allocate(set_neigh_info(conf)%threebody(ii)%dr_taper(3,cntr))
                    allocate(set_neigh_info(conf)%threebody(ii)%dr_taper_deriv(3,cntr))
                end if
               
                !* number of three-body terms centered on ii
                !feature_threebody_info(ii)%n = cntr ! DEP.
                set_neigh_info(conf)%threebody(ii)%n = cntr

                !* atomic number of central atom
                !feature_threebody_info(ii)%z_atom = data_sets(set_type)%configs(conf)%z(ii) ! DEP.
                set_neigh_info(conf)%threebody(ii)%z_atom = data_sets(set_type)%configs(conf)%z(ii)
               
                ! DEP vv
                !call dcopy(cntr,aniso_info%cos_ang,1,feature_threebody_info(ii)%cos_ang,1) 
                !feature_threebody_info(ii)%dr(:,:) = aniso_info%dr(:,1:cntr)
                !feature_threebody_info(ii)%z(:,:) = aniso_info%z(:,1:cntr)
                !feature_threebody_info(ii)%idx(:,:) = aniso_info%idx(:,1:cntr)
                !feature_threebody_info(ii)%dcos_dr(:,:,:) = aniso_info%dcos_dr(:,:,1:cntr)
                !feature_threebody_info(ii)%drdri(:,:,:) = aniso_info%drdri(:,:,1:cntr)
                !if (speedup_applies("threebody_rcut")) then
                !    feature_threebody_info(ii)%dr_taper(:,:) = aniso_info%dr_taper(:,1:cntr)
                !    feature_threebody_info(ii)%dr_taper_deriv(:,:) = aniso_info%dr_taper_deriv(:,&
                !            &1:cntr)
                !end if
                !! DEP ^^
                
                call dcopy(cntr,aniso_info%cos_ang,1,set_neigh_info(conf)%threebody(ii)%cos_ang,1)
                set_neigh_info(conf)%threebody(ii)%dr(:,:) = aniso_info%dr(:,1:cntr)
                set_neigh_info(conf)%threebody(ii)%z(:,:) = aniso_info%z(:,1:cntr)
                set_neigh_info(conf)%threebody(ii)%idx(:,:) = aniso_info%idx(:,1:cntr)
                set_neigh_info(conf)%threebody(ii)%dcos_dr(:,:,:) = aniso_info%dcos_dr(:,:,1:cntr)
                set_neigh_info(conf)%threebody(ii)%drdri(:,:,:) = aniso_info%drdri(:,:,1:cntr)
                if (speedup_applies("threebody_rcut")) then
                    set_neigh_info(conf)%threebody(ii)%dr_taper(:,:) = aniso_info%dr_taper(:,1:cntr)
                    set_neigh_info(conf)%threebody(ii)%dr_taper_deriv(:,:) = aniso_info%&
                            &dr_taper_deriv(:,1:cntr)
                end if

                
            end do !* end loop ii over local cell atoms
            
            deallocate(aniso_info%cos_ang)
            deallocate(aniso_info%dr)
            deallocate(aniso_info%z)
            deallocate(aniso_info%idx)
            deallocate(aniso_info%dcos_dr)
            deallocate(aniso_info%drdri)
            if (speedup_applies("threebody_rcut")) then
                deallocate(aniso_info%dr_taper)
                deallocate(aniso_info%dr_taper_deriv)
            end if
        end subroutine calculate_threebody_info

        real(8) function distance2(dr1,dr2)
            implicit none

            real(8),intent(in) :: dr1(1:3),dr2(1:3)

            distance2 = (dr1(1)-dr2(1))**2 + (dr1(2)-dr2(2))**2 + (dr1(3)-dr2(3))**2  
        end function distance2


        logical function int_in_intarray(idx,array,arg)
            implicit none

            integer,intent(in) :: idx,array(:)
            integer,intent(out) :: arg

            !* scratch
            integer :: ii
            logical :: ishere

            ishere = .false.
            arg = -1    ! NULL value, should not use if if ishere = False

            do ii=1,size(array)
                if (array(ii).eq.idx) then
                    !* idx is in array
                    ishere = .true.
                    
                    !* return index value where idx is found
                    arg = ii
                    exit
                end if
            end do

            int_in_intarray = ishere
        end function int_in_intarray

        real(8) function cos_angle(drij,drik,drij_mag,drik_mag)
            implicit none
            
            real(8),intent(in) :: drij(1:3),drik(1:3)
            real(8),intent(in) :: drij_mag,drik_mag

            !* assumes drij = rj - ri
            !          drik = rk - ri

            cos_angle = ddot(3,drij,1,drik,1) / (drij_mag*drik_mag)
        end function cos_angle

        subroutine error(routine,message)
            implicit none

            character(len=*),intent(in) :: routine,message

            character,dimension(1:len(routine)+26) :: header
            header(:) = "*"
            
            write(*,*) ''
            write(*,*) header
            write(*,*) 'error raised in routine : ',routine
            write(*,*) header
            write(*,*) ''
            write(*,*) 'Error : ',message
            call exit(0)
        end subroutine error

        subroutine deallocate_feature_deriv_info()
            implicit none

            integer :: set_type,conf,atm,ft
            
            do set_type=1,2
                do conf=1,data_sets(set_type)%nconf
                    do atm=1,data_sets(set_type)%configs(conf)%n
                        do ft=1,D
                            if (data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%n.ne.0) then
                                deallocate(data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%idx)
                                deallocate(data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%vec)
                            end if
                        end do
                    end do
                end do
            end do
        end subroutine

        subroutine copy_threebody_feature_info(feature_in,allocatable_feature)
            implicit none

            !* arg
            type(feature_info_threebody),intent(in) :: feature_in(:)
            type(feature_info_threebody),allocatable,intent(inout) :: allocatable_feature(:)

            !* scratch
            integer :: natm,ii

            natm = size(feature_in)

            allocate(allocatable_feature(natm))
            do ii=1,natm,1
                allocatable_feature(ii)%n = feature_in(ii)%n
                allocatable_feature(ii)%z_atom = feature_in(ii)%z_atom
                if (allocatable_feature(ii)%n.gt.0) then
                    allocate(allocatable_feature(ii)%cos_ang(allocatable_feature(ii)%n))
                    allocate(allocatable_feature(ii)%dr(3,allocatable_feature(ii)%n))
                    allocate(allocatable_feature(ii)%z(2,allocatable_feature(ii)%n))
                    allocate(allocatable_feature(ii)%idx(2,allocatable_feature(ii)%n))
                    allocate(allocatable_feature(ii)%dcos_dr(3,3,allocatable_feature(ii)%n))
                    allocate(allocatable_feature(ii)%drdri(3,6,allocatable_feature(ii)%n))

                    allocatable_feature(ii)%cos_ang(:) = feature_in(ii)%cos_ang(:)
                    allocatable_feature(ii)%dr(:,:) = feature_in(ii)%dr(:,:)
                    allocatable_feature(ii)%z(:,:) = feature_in(ii)%z(:,:)
                    allocatable_feature(ii)%idx(:,:) = feature_in(ii)%idx(:,:)
                    allocatable_feature(ii)%dcos_dr(:,:,:) = feature_in(ii)%dcos_dr(:,:,:)
                    allocatable_feature(ii)%drdri(:,:,:) = feature_in(ii)%drdri(:,:,:)
                end if
            end do
        end subroutine

        subroutine computeall_feature_scaling_constants(set_type)
            implicit none

            integer,intent(in) :: set_type

            !* scratch
            integer :: ii,conf,atm
            real(8) :: currentmax,currentmin

            do ii=1,feature_params%num_features,1
                if (feature_params%info(ii)%ftype.eq.featureID_StringToInt("atomic_number")) then
                    cycle
                end if

                currentmax = data_sets(set_type)%configs(1)%x(ii+1,1)
                currentmin = currentmax

                do conf=1,data_sets(set_type)%nconf,1
                    do atm=1,data_sets(set_type)%configs(conf)%n
                        if (data_sets(set_type)%configs(conf)%x(ii+1,atm).gt.currentmax) then
                            currentmax = data_sets(set_type)%configs(conf)%x(ii+1,atm)
                        end if
                        if (data_sets(set_type)%configs(conf)%x(ii+1,atm).lt.currentmin) then
                            currentmin = data_sets(set_type)%configs(conf)%x(ii+1,atm)
                        end if
                    end do
                end do
           
                if (abs(currentmax).lt.dble(10e-15)**2) then
                    !* all feature coordinates are zero!
                    feature_params%info(ii)%scl_cnst = 0.0d0
                    feature_params%info(ii)%add_cnst = 0.0d0
                else
                    !* scale to between [-1,1] (inclusive)
                    feature_params%info(ii)%scl_cnst = 2.0d0/(currentmax-currentmin)
                    feature_params%info(ii)%add_cnst = -1.0d0 -2.0d0*currentmin/(currentmax-&
                            &currentmin)
                end if
            end do
        end subroutine computeall_feature_scaling_constants

        subroutine scale_set_features(set_type,scale_derivatives)
            implicit none

            integer,intent(in) :: set_type
            logical,intent(in) :: scale_derivatives

            !* scratch
            integer :: conf
                
            do conf=1,data_sets(set_type)%nconf,1
                call scale_conf_features(set_type,conf,scale_derivatives)
            end do !* end loop over configurations
        end subroutine
        
        subroutine scale_conf_features(set_type,conf,scale_derivatives)
            implicit none

            integer,intent(in) :: set_type,conf
            logical,intent(in) :: scale_derivatives

            !* scratch
            integer :: ii,atm,jj
            real(8) :: cnst,cnst_add

            do ii=1,feature_params%num_features,1
                if (feature_params%info(ii)%ftype.eq.featureID_StringToInt("atomic_number")) then 
                    cycle
                end if

                cnst = feature_params%info(ii)%scl_cnst
                cnst_add = feature_params%info(ii)%add_cnst
                
                data_sets(set_type)%configs(conf)%x(ii+1,:) = &
                        &data_sets(set_type)%configs(conf)%x(ii+1,:)*cnst + cnst_add

                if (scale_derivatives) then
                    do atm=1,data_sets(set_type)%configs(conf)%n
                        if (data_sets(set_type)%configs(conf)%x_deriv(ii,atm)%n.eq.0) then
                            cycle
                        end if

                        do jj=1,data_sets(set_type)%configs(conf)%x_deriv(ii,atm)%n,1
                            data_sets(set_type)%configs(conf)%x_deriv(ii,atm)%vec(:,jj) = cnst*&
                                    &data_sets(set_type)%configs(conf)%x_deriv(ii,atm)%vec(:,jj)  
                        end do !* end loop over neighbours to atm
                    end do !* end loop over atoms
                end if 

            end do !* end loop over features
        end subroutine scale_conf_features

        subroutine Nbody_cutoff_parameters(N,rcut_array,fs_array)
            !* return array of rcut and fs parameters for all two body features
            
            implicit none

            !* args
            real(8),allocatable,intent(inout) :: rcut_array(:),fs_array(:)
            integer,intent(in) :: N

            !* scratch
            integer :: ft,ftype,cntr

            if ((N.ne.2).and.(N.ne.3)) then
                call error("Nbody_cutoff_parameters","incorrect param value N")
            end if

            cntr = 0

            do ft=1,feature_params%num_features,1
                ftype = feature_params%info(ft)%ftype

                if (N.eq.2) then
                    if (feature_IsTwoBody(ftype)) then
                        cntr = cntr + 1
                    end if
                else if (N.eq.3) then
                    if (feature_IsThreeBody(ftype)) then
                        cntr = cntr + 1
                    end if
                end if
            end do

            if (cntr.eq.0) then
                call error("Nbody_cutoff_parameters","no features found for given approximation")
            end if

            allocate(rcut_array(cntr))
            allocate(fs_array(cntr))

            cntr = 1
            do ft=1,feature_params%num_features,1
                ftype = feature_params%info(ft)%ftype

                if (N.eq.2) then
                    if (feature_IsTwoBody(ftype)) then
                        rcut_array(cntr) = feature_params%info(ft)%rcut
                        fs_array(cntr) = feature_params%info(ft)%fs
                        cntr = cntr + 1
                    end if
                else if (N.eq.3) then
                    if (feature_IsThreeBody(ftype)) then
                        rcut_array(cntr) = feature_params%info(ft)%rcut
                        fs_array(cntr) = feature_params%info(ft)%fs
                        cntr = cntr + 1
                    end if

                end if
            end do
        end subroutine Nbody_cutoff_parameters

        logical function performance_option_Nbody_rcut_applies(N)
            !* return True if all two body features share same rcut and fs
            use util, only : scalar_equal
            
            implicit none

            integer,intent(in) :: N

            !* scratch
            real(8),allocatable :: rcut_array(:),fs_array(:)
            real(8) :: minx,maxx
            logical :: tmp(1:2)

            !* fetch (rcut,fs) for all two body features
            call Nbody_cutoff_parameters(N,rcut_array,fs_array)

            tmp = .false.

            minx = minval(rcut_array)
            maxx = maxval(rcut_array)

            if (scalar_equal(minx,maxx,dble(1e-15),dble(1e-15),.false.)) then
                !* all rcuts are the same
                tmp(1) = .true.
            end if
            
            minx = minval(fs_array)
            maxx = maxval(fs_array)

            if (scalar_equal(minx,maxx,dble(1e-15),dble(1e-15),.false.)) then
                !* all fs' are the same
                tmp(2) = .true.
            end if

            performance_option_Nbody_rcut_applies = all(tmp)
        end function performance_option_Nbody_rcut_applies

        real(8) function get_Nbody_common_fs(N)
            !* return fs in common with all two body features
            implicit none

            !* args
            integer,intent(in) :: N

            !* scratch
            real(8) :: val = 0.0d0
            real(8),allocatable :: rcut_array(:),fs_array(:)

            if (N.eq.2) then
                if (speedup_applies("twobody_rcut")) then
                    call Nbody_cutoff_parameters(2,rcut_array,fs_array)

                    val = fs_array(1)
                else
                    call error("get_Nbody_common_fs","two body features have different fs")
                end if
            else if (N.eq.3) then
                if (speedup_applies("threebody_rcut")) then
                    call Nbody_cutoff_parameters(3,rcut_array,fs_array)

                    val = fs_array(1)
                else
                    call error("get_Nbody_common_fs","three body features have different fs")
                end if
            else
                call error("get_Nbody_common_fs","order of interaction not supported.")
            end if

            get_Nbody_common_fs = val
        end function get_Nbody_common_fs
end module        
