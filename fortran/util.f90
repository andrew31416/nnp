!==================================================================!
! NOTES                                                            !
! -----                                                            !
!   - for OpenMP and gnu, compile with --f90flags="-fopenmp -lgomp"!
!==================================================================!

module util
    implicit none

    contains
        integer function get_num_threads()
            !============================================================!
            ! Return the number of threads used in parallelised sections !
            ! of fortran, for given OMP_NUM_THREADS in unix environment  !
            !                                                            !
            ! To change this, export OMP_NUM_THREADS = <x> before        !
            ! executing Python                                           !
            !============================================================!

            !$ use omp_lib

            implicit none

            get_num_threads = omp_get_max_threads()
        
        end function get_num_threads

        subroutine frac_to_cart(frac,cell,cart)
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
        end subroutine frac_to_cart

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

            call frac_to_cart(tmp,cell,cart)
        end subroutine

        subroutine get_ultracellf90(localfrac,localcart,localcell,localZ,natm,rcut,maxsize,&
                &ultracart,ultraidx,ultraZ)
            !============================================================!
            ! return ultracell of cartesian positions from local         !
            ! cartesian positions                                        !
            !
            ! localcell(i,j) = ith cartesian coordinate of jth vector    !
            !============================================================!

            implicit none

            integer,intent(in) :: natm,maxsize
            real(8),intent(in) :: localfrac(1:3,1:natm),localcell(1:3,1:3)
            real(8),intent(in) :: rcut,localZ(1:natm),localcart(1:3,1:natm)
            real(8),allocatable,intent(inout) :: ultracart(:,:),ultraZ(:)
            integer,allocatable,intent(inout) :: ultraidx(:)

            !* scratch
            integer :: layer,maxiter,atomii,atomjj,ii,jj,kk,cntr
            integer :: ultranatm,tmpidx(1:maxsize)
            real(8) :: fracii(1:3),cartii(1:3),cartjj(1:3),rcut2
            real(8) :: tmpcart(1:3,1:maxsize),tmpZ(1:maxsize)
            logical :: pairfound

            ultranatm = 0

            rcut2 = rcut**2
            maxiter = 10

            do layer=1,maxiter
                cntr = 0

                do atomii=1,natm
                    !* fractional coordinate in local cell
                    fracii(1:3) = localfrac(1:3,atomii)

                    !* search over all layers
                    do ii=-layer,layer
                        do jj=-layer,layer
                            do kk=-layer,layer
                
                                !* only if one atom pair with local cell is
                                !* found do we add this image of atomii
                                pairfound = .false.

                                do atomjj=1,natm
                                    !* cartesians of atomjj
                                    cartjj(:) = localcart(:,atomjj)
                                    
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
                                    tmpZ(ultranatm) = localZ(atomii) 
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
                
                call frac_to_cart(localfrac(1:3,atomii),localcell,cartii)
                
                !* store cartesian coordinates
                tmpcart(1:3,ultranatm) = cartii(1:3)

                !* store corresponding local cell idx
                tmpidx(ultranatm) = atomii

                !* store atomic number 
                tmpZ(ultranatm) = localZ(atomii)
            end do
            
            allocate (ultracart(3,ultranatm))
            allocate (ultraidx(ultranatm))
            allocate (ultraZ(ultranatm))
            ultracart(1:3,1:ultranatm) = tmpcart(1:3,1:ultranatm)
            ultraidx(1:ultranatm) = tmpidx(1:ultranatm)
            ultraZ(1:ultranatm) = tmpZ(1:ultranatm)

        end subroutine get_ultracellf90

        
        subroutine atomatom_distances(natm,dr2,rcut,displacement,allocated_memory)
            !* return array of atom-atom distances within specified cut off
            !* radius

            implicit none

            integer,intent(in) :: natm(1:2)
            real(8),intent(in) :: dr2(1:natm(2),1:natm(1)),rcut
            real(8),intent(out),allocatable :: displacement(:)
            logical,intent(out) :: allocated_memory
    
            !* scratch
            real(8) :: tmp_disp(1:natm(1)*natm(2)),rcut2
            integer :: cntr,ii,jj

            cntr = 0
            rcut2 = rcut**2
            allocated_memory = .false.

            do ii=1,natm(1),1
                do jj=1,natm(2),1
                    if ( (dr2(jj,ii).le.rcut2).and.(dr2(jj,ii).gt.0.000001d0) ) then
                        !* account for local atom being contained in ultra cell atoms
                        cntr = cntr + 1
                        tmp_disp(cntr) = sqrt(dr2(jj,ii))
                   end if
                end do
            end do

            if (cntr.gt.0) then
                allocate(displacement(cntr))
                allocated_memory = .true.
                displacement(1:cntr) = tmp_disp(1:cntr)
            end if

        end subroutine atomatom_distances

        logical function angular_info(atom,localpos,ultrapos,localZ,ultraZ,Natm,rcut,angles)
            !* return (cos(dtheta_{ijk}),dr_{ij},dr_{ik},Z_i,Z_j,Z_k,dr_{jk}) for every
            !* triplet ijk within cut off radius of each other

            implicit none

            integer,intent(in) :: atom,Natm(1:2)
            real(8),intent(in) :: localpos(1:3,1:Natm(1)),ultrapos(1:3,1:Natm(2))
            real(8),intent(in) :: rcut,localz(1:Natm(1)),ultraz(1:Natm(2))
            real(8),allocatable,intent(out) :: angles(:,:)

            !* scratch
            integer :: jj,kk,ll,cntr
            real(8) :: drij_vec(1:3),drik_vec(1:3),drij2,drik2,drjk2
            real(8) :: rcut2,tmp_array(1:7,1000),pos(1:3),Zj,Zk,zatom

            rcut2 = rcut**2
    
            !* number of valid triplets found
            cntr = 0
   
            !* cartesians of central atom in triplet
            pos(1:3) = localpos(1:3,atom)

            !* atomic number
            zatom = localz(atom)

            do jj=1,Natm(2),1
                drij_vec(:) = ultrapos(:,jj)-pos(:)

                drij2 = drij_vec(1)**2+drij_vec(2)**2+drij_vec(3)**2

                if ( (drij2.gt.rcut2).or.(drij2.le.0.00001d0**2) ) then
                    !* do not want to pair with oneself
                    cycle
                end if

                Zj = ultraz(jj)
                
                do kk=1,Natm(2),1
                    if (jj.eq.kk) then
                        !* must have 3 distinct atoms
                        cycle
                    end if

                    drik_vec(:) = ultrapos(:,kk)-pos(:)

                    drik2 = drik_vec(1)**2+drik_vec(2)**2+drik_vec(3)**2

                    if ( (drik2.gt.rcut2).or.(drik2.le.0.00001d0**2) ) then
                        !* do not want to pair with oneself
                        cycle
                    end if
                   
                    Zk = ultraz(kk)

                    !* check neighbours are within rcut of one another
                    drjk2 = 0.0d0
                    
                    do ll=1,3,1
                        drjk2 = drjk2 + (drij_vec(ll)-drik_vec(ll))**2
                    end do
                    
                    if (drjk2.gt.rcut2) then
                        cycle
                    end if

                    !* if here then triplet is valid
                    cntr = cntr + 1

                    tmp_array(1,cntr) = sum(drij_vec*drik_vec)/(sqrt(drij2)*sqrt(drik2)) 
                    tmp_array(2,cntr) = sqrt(drij2)  
                    tmp_array(3,cntr) = sqrt(drik2)
                    tmp_array(4,cntr) = Zatom
                    tmp_array(5,cntr) = Zj
                    tmp_array(6,cntr) = Zk
                    tmp_array(7,cntr) = sqrt(drjk2)
                    
                    !!* double count unique triplets
                    !cntr = cntr + 1

                    !tmp_array(1,cntr) = tmp_array(1,cntr-1)  
                    !tmp_array(2,cntr) = tmp_array(3,cntr-1)
                    !tmp_array(3,cntr) = tmp_array(2,cntr-1)
                    !tmp_array(4,cntr) = tmp_array(4,cntr-1)
                    !tmp_array(5,cntr) = tmp_array(6,cntr-1)
                    !tmp_array(6,cntr) = tmp_array(5,cntr-1)
                    !tmp_array(7,cntr) = tmp_array(7,cntr-1)
                end do
            end do

            !* now dynamically allocate array
            if (cntr.gt.0) then
                allocate(angles(7,cntr))

                angles(:,:) = tmp_array(:,1:cntr)

                angular_info = .true.
            else 
                angular_info = .false.
            end if
        end function angular_info

        subroutine angular_info_structure(localpos,ultrapos,localZ,ultraZ,Natm,rcut,&
                &buffer_size,angles_structure)
            !* return (cos(dtheta_{ijk}),dr_{ij},dr_{ik},Z_i,Z_j,Z_k,dr_{jk}) for every
            !* triplet ijk within cut off radius of each other

            implicit none

            integer,intent(in) :: Natm(1:2),buffer_size
            real(8),intent(in) :: localpos(1:3,1:Natm(1)),ultrapos(1:3,1:Natm(2))
            real(8),intent(in) :: rcut,localz(1:Natm(1)),ultraz(1:Natm(2))
            real(8),allocatable,intent(out) :: angles_structure(:,:)

            !* scratch
            integer :: atom,num_triplets(1:Natm(1)),cntr,offset
            real(8),allocatable :: angles(:,:),tmp_angles_structure(:,:,:)

            allocate(tmp_angles_structure(7,buffer_size,Natm(1)))

            cntr = 0

            do atom=1,Natm(1),1
                if (angular_info(atom,localpos,ultrapos,localZ,ultraZ,Natm,rcut,angles)) then
                    num_triplets(atom) = int(float(size(angles))/7.0)
                    
                    if ( num_triplets(atom).gt.buffer_size) then
                        write(*,*) 'buffer size of',buffer_size,'is inadequate for ',&
                                &num_triplets(atom),'must increase'
                        call exit(0)
                    end if
                    
                    tmp_angles_structure(:,1:num_triplets(atom),atom) = angles(:,:)

                    cntr = cntr + num_triplets(atom)

                    deallocate(angles)
                end if
                
            end do

            allocate(angles_structure(7,cntr))

            do atom=1,Natm(1),1
                if (atom.eq.1) then
                    offset = 1
                else
                    offset = sum(num_triplets(1:atom-1)) + 1
                end if
                
                angles_structure(:,offset:offset+num_triplets(atom)-1) = &
                        &tmp_angles_structure(:,1:num_triplets(atom),atom)
            end do
        end subroutine angular_info_structure


        subroutine multiple_determinant(num_components,matrices,dets)
            implicit none
            
            integer,intent(in) :: num_components
            real(8),intent(in) :: matrices(1:3,1:3,1:num_components)
            real(8),intent(out) :: dets(1:num_components)

            !* scratch
            integer :: ii

            do ii=1,num_components,1
                dets(ii) = matrices(1,1,ii)*matrices(2,2,ii)*matrices(3,3,ii)
                dets(ii) = dets(ii) + matrices(1,2,ii)*matrices(2,3,ii)*matrices(3,1,ii)
                dets(ii) = dets(ii) + matrices(1,3,ii)*matrices(2,1,ii)*matrices(3,2,ii)
                dets(ii) = dets(ii) - matrices(1,1,ii)*matrices(2,3,ii)*matrices(3,2,ii)
                dets(ii) = dets(ii) - matrices(1,2,ii)*matrices(2,1,ii)*matrices(3,3,ii)
                dets(ii) = dets(ii) - matrices(1,3,ii)*matrices(2,2,ii)*matrices(3,1,ii)
            end do

        end subroutine multiple_determinant
end module util
