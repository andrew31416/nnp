module feature_util
    use config
    use feature_config

    implicit none

    !* blas/lapack
    external :: dcopy
    external :: dgemm
    real(8),external :: ddot

    contains

        subroutine invert_matrix(mtrx_in,mtrx_out)
            implicit none
            
            real(8),intent(in) :: mtrx_in(1:3,1:3)
            real(8),intent(inout) :: mtrx_out(1:3,1:3)

            real(8) :: det
            
            mtrx_out(1,1) = mtrx_in(2,2)*mtrx_in(3,3) - mtrx_in(3,2)*mtrx_in(2,3)
            mtrx_out(2,1) = mtrx_in(2,3)*mtrx_in(3,1) - mtrx_in(2,1)*mtrx_in(3,3)
            mtrx_out(3,1) = mtrx_in(2,1)*mtrx_in(3,2) - mtrx_in(3,1)*mtrx_in(2,2) 

            mtrx_out(1,2) = mtrx_in(3,2)*mtrx_in(1,3) - mtrx_in(1,2)*mtrx_in(3,3)
            mtrx_out(2,2) = mtrx_in(1,1)*mtrx_in(3,3) - mtrx_in(3,1)*mtrx_in(1,3)
            mtrx_out(3,2) = mtrx_in(3,1)*mtrx_in(1,2) - mtrx_in(1,1)*mtrx_in(3,2)

            mtrx_out(1,3) = mtrx_in(1,2)*mtrx_in(2,3) - mtrx_in(2,2)*mtrx_in(1,3)
            mtrx_out(2,3) = mtrx_in(2,1)*mtrx_in(1,3) - mtrx_in(1,1)*mtrx_in(2,3)
            mtrx_out(3,3) = mtrx_in(1,1)*mtrx_in(2,2) - mtrx_in(2,1)*mtrx_in(1,2)

            det = mtrx_in(1,1)*mtrx_out(1,1) + mtrx_in(1,2)*mtrx_out(2,1) + &
                    &mtrx_in(1,3)*mtrx_out(3,1)

            mtrx_out(:,:) = mtrx_out(:,:)/det                  

        end subroutine

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
                    if ( (ftype.eq.featureID_StringToInt("acsf_behler-g1")).or.&
                    &(ftype.eq.featureID_StringToInt("acsf_behler-g2")).or.&
                    &(ftype.eq.featureID_StringToInt("acsf_normal-iso")) ) then
                    !if ( (ftype.eq.0).or.(ftype.ne.1) ) then
                        !* all isotropic features
                        tmprcut(ii) = tmpr
                    else
                        tmprcut(ii) = -1.0d0
                    end if
                else if (arg.eq.2) then
                    if ( (ftype.eq.featureID_StringToInt("acsf_behler-g4")).or.&
                    &(ftype.eq.featureID_StringToInt("acsf_behler-g5")).or.&
                    &(ftype.eq.featureID_StringToInt("acsf_normal-ani")) ) then
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
            integer :: ii,ftype

            found_some = .false.

            do ii=1,feature_params%num_features,1
                ftype = feature_params%info(ii)%ftype

                if ( (ftype.eq.2).or.(ftype.eq.4) ) then
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

            implicit none

            real(8),intent(in) :: ultracart(:,:),ultraz(:)
            integer,intent(in) :: ultraidx(:),set_type,conf

            !* scratch
            integer :: dim(1:1),ii,jj,cntr
            real(8) :: rcut2,dr2,rtol2
            real(8) :: drii(1:3),drjj(1:3)

            !* dim(1) = number atoms in ultra cell
            dim = shape(ultraidx)

            !* max isotropic interaction cut off
            rcut2 = maxrcut(1)**2

            !* min distance between 2 different atoms allowed
            rtol2 = (0.0000001)**2

            !* info for each atom
            allocate(feature_isotropic(data_sets(set_type)%configs(conf)%n))

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


                !* allocate neighbour mem
                allocate(feature_isotropic(ii)%dr(cntr))
                allocate(feature_isotropic(ii)%idx(cntr))
                allocate(feature_isotropic(ii)%z(cntr))
                allocate(feature_isotropic(ii)%drdri(3,cntr))
               
                !* number of neighbours 
                feature_isotropic(ii)%n = cntr
                
                !* atomic number of central atom in interaction
                feature_isotropic(ii)%z_atom = data_sets(set_type)%configs(conf)%z(ii)

                cntr = 1
                do jj=1,dim(1),1
                    drjj(:) = ultracart(:,jj)

                    dr2 = distance2(drii,drjj) 

                    if ( (dr2.lt.rtol2).or.(dr2.gt.rcut2) ) then
                        !* same atom or beyond cut off
                        cycle
                    else 
                        !* atom-atom distance
                        feature_isotropic(ii)%dr(cntr) = sqrt(dr2)

                        !* local cell identifier of neighbour cntr
                        feature_isotropic(ii)%idx(cntr) = ultraidx(jj)

                        !* d rij / drj 
                        feature_isotropic(ii)%drdri(:,cntr) = (drjj(:) - drii(:)) / &
                                &feature_isotropic(ii)%dr(cntr)
                        
                        !* Z of neighbour cntr
                        feature_isotropic(ii)%z(cntr) = ultraz(jj) 
                        cntr = cntr + 1    
                    end if
                end do


            end do
        end subroutine calculate_twobody_info
        
        
        subroutine calculate_threebody_info(set_type,conf,ultracart,ultraz,ultraidx)
            !===============================================================!
            !* calculate isotropic atom-atom distances and derivatives     *!
            !===============================================================!

            implicit none

            real(8),intent(in) :: ultracart(:,:),ultraz(:)
            integer,intent(in) :: ultraidx(:),set_type,conf

            !* scratch
            integer :: dim(1:1),ii,jj,kk,zz,cntr
            real(8) :: rcut2,dr2ij,dr2ik,dr2jk,rtol2
            real(8) :: rii(1:3),rjj(1:3),rkk(1:3),drij,drik,drjk
            real(8) :: drij_vec(1:3),drik_vec(1:3),drjk_vec(1:3)
            real(8) :: sign_ij(1:3),sign_ik(1:3)
            integer :: maxbuffer
            type(feature_info_threebody) :: aniso_info
real(8) :: tmp

            !* dim(1) = number atoms in ultra cell
            dim = shape(ultraidx)

            !* max anisotropic interaction cut off
            rcut2 = maxrcut(2)**2
            
            !* min distance between 2 different atoms allowed
            rtol2 = (0.0000001)**2

            !* max number of assumed 3-body terms per atom
            maxbuffer = 50000


            !* structure for all three body info associated with structure
            allocate(feature_threebody_info(data_sets(set_type)%configs(conf)%n))
            
            allocate(aniso_info%cos_ang(maxbuffer))
            allocate(aniso_info%dr(3,maxbuffer))
            allocate(aniso_info%z(2,maxbuffer))
            allocate(aniso_info%idx(2,maxbuffer))
            allocate(aniso_info%dcos_dr(3,3,maxbuffer))
            allocate(aniso_info%drdri(3,6,maxbuffer))

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
                        
                        if ( (dr2ik.lt.rtol2).or.(dr2ik.gt.rcut2).or.(dr2jk.gt.rcut2) ) then
                            cycle
                        end if

                        drik = sqrt(dr2ik)
                        drjk = sqrt(dr2jk)
                        drik_vec = rkk - rii
                        drjk_vec = rkk - rjj

                        !* have found a three-body term
                        cntr = cntr + 1  
                        if (cntr.gt.maxbuffer) then
                            call error("calculate_threebody_info","value of maxbuffer too small, increase size")
                        end if
                            
                       
                        !* cos(dtheta_{ijk}) 
                        aniso_info%cos_ang(cntr) =  cos_angle(rjj-rii,rkk-rii,drij,drik)

                        !* displacements
                        aniso_info%dr(1,cntr) = drij    ! central vs. jj
                        aniso_info%dr(2,cntr) = drik    ! central vs. kk
                        aniso_info%dr(3,cntr) = drjk    ! jj vs. kk
           
                        !* atomic number
                        aniso_info%z(1,cntr) = ultraz(jj)
                        aniso_info%z(2,cntr) = ultraz(kk)
                   
                        !* local cell identifier
                        aniso_info%idx(1,cntr) = ultraidx(jj)
                        aniso_info%idx(2,cntr) = ultraidx(kk)
                  
                        !---------------------------------!
                        !* atom-atom distance derivative *!
                        !---------------------------------!
tmp = 0.0d0
aniso_info%drdri(:,:,cntr) = 1.0d0/tmp
sign_ij(:) = 1.0d0/tmp          
sign_ik(:) = 1.0d0/tmp          
                 
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

                        ! remember that d |rj-ri| / dri = - d |rj-ri| / drj
                   
                        !---------------------!
                        !* cosine derivative *!
                        !---------------------!
                      
                        ! j=1 , k=2 , i=3
                        do zz=1,3,1
                            aniso_info%dcos_dr(:,zz,cntr) = 1.0d0/(drij*drik) * ( drij_vec*(sign_ik(zz)-&
                                    &aniso_info%cos_ang(cntr)*sign_ij(zz)*drik/drij) + &
                                    &drik_vec*(sign_ij(zz) - aniso_info%cos_ang(cntr)*sign_ik(zz)*drij/drik) )
                        end do
                        
                        !! dcos_{ijk} / drj
                        !aniso_info%dcos_dr(:,1,cntr) = 1.0d0/(drij*drik) * ( aniso_info%cos_ang(cntr) * ( &
                        !        &drik*aniso_info%drdri(:,1,cntr) + drij*aniso_info%drdri(:,4,cntr) ) + &
                        !        &drij_vec*aniso_info%drdri(:,4,cntr) + drik_vec*aniso_info%drdri(:,1,cntr) )
                        !
                        !
                        !!! dcos_{ijk} / drk
                        !aniso_info%dcos_dr(:,2,cntr) = 1.0d0/(drij*drik) * ( aniso_info%cos_ang(cntr) * ( &
                        !        &drik*aniso_info%drdri(:,2,cntr) + drij*aniso_info%drdri(:,3,cntr) ) + &
                        !        &drij_vec*aniso_info%drdri(:,3,cntr) + drik_vec*aniso_info%drdri(:,2,cntr) )
                        !
                        !
                        !!! dcos_{ijk} / dri
                        !aniso_info%dcos_dr(:,3,cntr) = 1.0d0/(drij*drik) * ( aniso_info%cos_ang(cntr) * ( &
                        !        &-drik*aniso_info%drdri(:,1,cntr) - drij*aniso_info%drdri(:,3,cntr) ) - &
                        !        &drij_vec*aniso_info%drdri(:,3,cntr) - drik_vec*aniso_info%drdri(:,3,cntr) )
                        
                    end do !* end loop kk over second neighbours
                end do !* end loop jj over first neighbours
                

                !* now copy into 
                allocate(feature_threebody_info(ii)%cos_ang(cntr))
                allocate(feature_threebody_info(ii)%dr(3,cntr))
                allocate(feature_threebody_info(ii)%z(2,cntr))
                allocate(feature_threebody_info(ii)%idx(2,cntr))
                allocate(feature_threebody_info(ii)%dcos_dr(3,3,cntr))
                allocate(feature_threebody_info(ii)%drdri(3,6,cntr))
               
                !* number of three-body terms centered on ii
                feature_threebody_info(ii)%n = cntr

                !* atomic number of central atom
                feature_threebody_info(ii)%z_atom = data_sets(set_type)%configs(conf)%z(ii)
               
                call dcopy(cntr,aniso_info%cos_ang,1,feature_threebody_info(ii)%cos_ang,1)                
                feature_threebody_info(ii)%dr(:,:) = aniso_info%dr(:,1:cntr)
                feature_threebody_info(ii)%z(:,:) = aniso_info%z(:,1:cntr)
                feature_threebody_info(ii)%idx(:,:) = aniso_info%idx(:,1:cntr)
                feature_threebody_info(ii)%dcos_dr(:,:,:) = aniso_info%dcos_dr(:,:,1:cntr)
                feature_threebody_info(ii)%drdri(:,:,:) = aniso_info%drdri(:,:,1:cntr)
                
            end do !* end loop ii over local cell atoms
            
            deallocate(aniso_info%cos_ang)
            deallocate(aniso_info%dr)
            deallocate(aniso_info%z)
            deallocate(aniso_info%idx)
            deallocate(aniso_info%dcos_dr)
            deallocate(aniso_info%drdri)
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
end module        
