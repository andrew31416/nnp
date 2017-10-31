module feature_util
    use config
    use feature_config

    implicit none

    contains

        subroutine invert_matrix(mtrx_in,mtrx_out)
            implicit none

            real(8),intent(in) :: mtrx_in(1:3,1:3)
            real(8),intent(inout) :: mtrx_out(1:3,1:3)

            real(8) :: det

            mtrx_out(1,1) = mtrx_in(2,2)*mtrx_in(3,3) - mtrx_in(3,2)*mtrx_in(2,3)
            mtrx_out(2,1) = mtrx_in(2,3)*mtrx_in(3,1) - mtrx_in(2,1)*mtrx_in(3,3)
            mtrx_out(3,1) = mtrx_in(2,1)*mtrx_in(3,2) - mtrx_in(3,1)*mtrx_in(2,2) 

            mtrx_out(1,2) = mtrx_in(3,2)*mtrx_in(1,3) - mtrx_in(2,1)*mtrx_in(3,3)
            mtrx_out(2,2) = mtrx_in(1,1)*mtrx_in(3,3) - mtrx_in(3,1)*mtrx_in(1,3)
            mtrx_out(3,2) = mtrx_in(3,1)*mtrx_in(1,2) - mtrx_in(1,1)*mtrx_in(3,2)

            mtrx_out(1,3) = mtrx_in(1,2)*mtrx_in(2,3) - mtrx_in(2,2)*mtrx_in(1,3)
            mtrx_out(2,3) = mtrx_in(2,1)*mtrx_in(1,3) - mtrx_in(1,1)*mtrx_in(1,3)
            mtrx_out(3,3) = mtrx_in(1,1)*mtrx_in(2,2) - mtrx_in(2,1)*mtrx_in(1,2)

            det = mtrx_in(1,1)*mtrx_out(1,1) + mtrx_in(2,2)*mtrx_out(2,1) + &
                    &mtrx_in(1,3)*mtrx_out(3,1)

            mtrx_out(:,:) = mtrx_out(:,:)/det                    
        end subroutine

        subroutine matrix_vec_mult(frac,cell,cart)
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

            call matrix_vec_mult(tmp,cell,cart)
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
                    call matrix_vec_mult(cartii,invcell,fracii)

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
                    if (ftype.eq.0) then
                        tmprcut(ii) = -1.0d0
                    else
                        !* all features
                        tmprcut(ii) = tmpr
                    end if
                else if (arg.eq.1) then
                    if ( (ftype.eq.0).or.(ftype.ne.1) ) then
                        !* all isotropic features
                        tmprcut(ii) = -1.0d0
                    else
                        tmprcut(ii) = tmpr
                    end if
                else if (arg.eq.2) then
                    if ( (ftype.eq.0).or.(ftype.ne.2) ) then
                        tmprcut(ii) = -1.0d0
                    else
                        tmprcut(ii) = tmpr
                    end if
                end if
            end do

            tmpr = maxval(tmprcut)
            deallocate(tmprcut)
    
            maxrcut = tmpr
        end function maxrcut

        subroutine calculate_isotropic_info(set_type,conf,ultracart,ultraz,ultraidx)
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

                cntr = 1
                do jj=1,dim(1),1
                    drjj(:) = ultracart(:,jj)

                    dr2 = distance2(drii,drjj) 

                    if ( (dr2.lt.rtol2).or.(dr2.gt.rcut2) ) then
                        !* same atom or beyond cut off
                        cycle
                    else 
                        feature_isotropic(ii)%dr(cntr) = sqrt(dr2)
                        feature_isotropic(ii)%idx(cntr) = ultraidx(jj)
                        feature_isotropic(ii)%drdri(:,cntr) = drjj(:) - drii(:)
                        feature_isotropic(ii)%z(cntr) = ultraz(jj) 
                        cntr = cntr + 1    
                    end if
                end do


            end do
        end subroutine calculate_isotropic_info

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
end module        
