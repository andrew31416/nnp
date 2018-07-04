module sedc
    implicit none


    contains
        real(8) function damping(ReffA,ReffB,drAB,d,sR)
            !===================================================================!
            ! RAB = ReffA + ReffB
            ! drAB = distance between both atoms
            ! d,sR= parameters
            !
            ! damping = 1 / ( 1 + exp(-d[drAB/RAB -1]) )
            !===================================================================!
            
            implicit none

            real(8),intent(in) :: ReffA,ReffB,drAB,d,sR
        
            !* scratch
            real(8) :: RAB

            RAB = ReffA + ReffB

            damping = 1.0d0 / (1.0d0 + exp( -d*(drAB/(sR*RAB) - 1.0d0) ))
        end function damping
        
        real(8) function calc_c6(Reff,Rfree,C6free)
            implicit none

            real(8),intent(in) :: Reff,Rfree,C6free

            calc_c6 = (Reff/Rfree)**6 * C6free
        end function calc_c6
        
        real(8) function feature_normal(dr2,Zultra,Natm,Zatm,rcut,mean,normal_precision,&
                &zi_pow_const,zj_pow_const)
            !=======================================================!
            !                                                       ! 
            ! ( Lambda / (2 pi) )^0.5 * taper(drij,rcut) *          !
            ! (Zi^zi_pow_const) * (Zj^zj_pow_const) * exp( -0.5 *   !
            ! Lambda * (drij-mu)^2 )                                !
            !=======================================================!

            use rvm_basis, only : wrapped_smooth
            
            implicit none
            
            !* arguments
            integer,intent(in) :: Natm
            real(8),intent(in) :: Zultra(1:Natm),dr2(1:Natm),rcut,zj_pow_const
            real(8),intent(in) :: Zatm,normal_precision,mean,zi_pow_const
    

            !* scratch
            integer :: ii
            real(8) :: rcut2,tmp,rtol,gammavalue
            
            rcut2 = rcut**2
            gammavalue = 0.0d0
        
            !* pos could be duplicated in ultrapos
            rtol = 0.00000001d0
            
            do ii=1,Natm
                !* distance**2 between both atoms
                tmp = dr2(ii)
                
                if ( (tmp.lt.rtol).or.(tmp.gt.rcut2) ) then
                    cycle
                end if
                
                tmp = sqrt(tmp)
                
                gammavalue = gammavalue + ((Zultra(ii)+1.0d0)**zj_pow_const)*&
                        &sqrt(0.5d0*normal_precision/3.141592653589793d0)&
                        &*exp(-0.5d0*(tmp-mean)**2 * normal_precision)*&
                        &wrapped_smooth(0.2d0,rcut,.true.,tmp)
            
            end do
            
            feature_normal = gammavalue*((Zatm+1.0d0)**zi_pow_const)
        end function

        real(8) function gamma_gaussian(dr2,Zultra,Natm,invk2,Zk,rcut)
            use rvm_basis, only : wrapped_smooth
            
            implicit none
            
            !* arguments
            integer,intent(in) :: Natm
            real(8),intent(in) :: Zultra(1:Natm),dr2(1:Natm),rcut
            real(8),intent(in) :: invk2,Zk

            !* scratch
            integer :: ii
            real(8) :: rcut2,tmp,rtol,gammavalue
            
            rcut2 = rcut**2
            gammavalue = 0.0d0
        
            !* pos could be duplicated in ultrapos
            rtol = 0.00000001d0

            do ii=1,Natm
                !* distance**2 between both atoms
                tmp = dr2(ii)
                
                if ( (tmp.lt.rtol).or.(tmp.gt.rcut2) ) then
                    cycle
                end if
                
                tmp = sqrt(tmp)

                gammavalue = gammavalue + (Zultra(ii)**Zk)*exp(-0.5d0*tmp*invk2)*&
                        &wrapped_smooth(0.2d0,rcut,.true.,tmp)
            end do
            
            gamma_gaussian = gammavalue

        end function gamma_gaussian

        real(8) function nearest_image_distance(dfrac,cell)
            implicit none

            real(8),intent(inout) :: dfrac(1:3)
            real(8),intent(in) :: cell(1:3,1:3)

            integer :: ii,jj
            real(8) :: dr(1:3)

            dr(:) = 0.0d0

            !* classic 'dumb' nearest neighbour
            do ii=1,3
                dfrac(ii) = dfrac(ii) -nint(dfrac(ii))
            end do

            do ii=1,3
                do jj=1,3
                    dr(ii) = dr(ii) + cell(ii,jj)*dfrac(jj)
                end do
            end do
            nearest_image_distance = sqrt(dr(1)**2+dr(2)**2+dr(3)**2)
        end function nearest_image_distance

        real(8) function pair_c6(c6AA,c6BB,alphaA,alphaB)
            implicit none

            real(8),intent(in) :: c6AA,c6BB,alphaA,alphaB

            pair_c6 = (2.0d0*c6AA*c6BB)/( (alphaB/alphaA)*c6AA + (alphaA/alphaB)*c6BB )
        end function

        real(8) function energy_TS(localfrac,cell,natm,Reff,Rfree,c6free,alpha_free,d,sR)
            implicit none

            integer,intent(in) :: natm
            real(8),intent(in) :: localfrac(1:3,1:natm),cell(1:3,1:3)
            real(8),intent(in) :: Reff(1:natm),Rfree(1:natm),c6free(1:natm)
            real(8),intent(in) :: alpha_free(1:natm),d,sR

            !* scratch
            integer :: ii,jj
            real(8) :: drij,fii(1:3),c6eff(1:natm),c6ij,c6eff_ii
            real(8) :: energy,Reff_ii,alpha_ii,dfrac(1:3),alpha_eff(1:natm)

            energy = 0.0d0

            !* get effective C6 coefficients
            call allc6eff(Reff,Rfree,c6free,natm,c6eff)
            
            !* get effecitve atomic polarizability
            call all_alpha_eff(alpha_free,Reff,Rfree,natm,alpha_eff)

            do ii=1,natm
                !* stop page thrashing for large natm
                fii(:) = localfrac(:,ii)
                c6eff_ii = c6eff(ii)
                Reff_ii = Reff(ii)
                alpha_ii = alpha_eff(ii)

                do jj=1+1,natm
                    dfrac(:) = fii(:) - localfrac(:,jj)

                    !* simplest nearest image distance
                    drij = nearest_image_distance(dfrac,cell)
                
                    !* effective pair c6 coefficient
                    c6ij = pair_c6(c6eff_ii,c6eff(jj),alpha_ii,alpha_eff(jj))

                    energy = energy - damping(Reff_ii,Reff(jj),drij,d,sR)*c6ij/(drij**6)
                end do
            end do
            energy_TS = energy
        end function energy_TS

        subroutine displacement_matrix(localpos,ultrapos,Natm,dr2)
            !===================================================================!
            ! get matrix of squared displacements between local atoms and       !
            ! ultracell atoms                                                   !
            !===================================================================!
            
            implicit none
    
            integer,intent(in) :: Natm(1:2)
            real(8),intent(in) :: localpos(1:3,1:Natm(1)),ultrapos(1:3,1:Natm(2))
            real(8),intent(out) :: dr2(1:Natm(2),1:Natm(1))

            !* scratch
            integer ii,jj
            real(8) :: pos(1:3)

            do ii=1,Natm(1)
                pos(1:3) = localpos(1:3,ii)

                do jj=1,Natm(2)
                    dr2(jj,ii) = (pos(1)-ultrapos(1,jj))**2+(pos(2)-ultrapos(2,jj))**2+(pos(3)-ultrapos(3,jj))**2
                end do
            end do

        end subroutine displacement_matrix

        subroutine observables_singlestructure(Reff,Rvac,obs)
            !===================================================================!
            ! generate a list of observables for vdw radius in vacuum, Rvac and !
            ! vdw_radius in environment, Reff.                                  !
            !                                                                   !
            ! Output                                                            !
            ! ------                                                            !
            !   - Reff(:) - Rvac(:)                                             !
            !===================================================================!

            implicit none
            
            !* arguments
            real(8),intent(in) :: Reff(:),Rvac(:)
            real(8),intent(out) :: obs(:)

            obs(:) = Reff(:) - Rvac(:)
        
        end subroutine observables_singlestructure            

        subroutine allc6eff(Reff,Rfree,C6free,n,C6eff)
            !===================================================================!
            ! calculate effective C6 coefficients for all effective vdw radii,  !
            ! free (vacuum) radii and free C6 coefficients                      !
            !===================================================================!

            implicit none
            
            integer,intent(in) :: n
            real(8),intent(in) :: Reff(1:n),Rfree(1:n),C6free(1:n)
            real(8),intent(out) :: C6eff(1:n)

            !* scratch
            integer :: ii
            
            do ii=1,n
                C6eff(ii) = calc_c6(Reff(ii),Rfree(ii),C6free(ii))
            end do
        end subroutine allc6eff

        subroutine all_alpha_eff(alpha_free,Reff,Rfree,natm,alpha_eff)
            !===================================================================!
            ! calculate effective polarizability, scaled by ratio of effective  !
            ! to free volumes                                                   !
            !===================================================================!

            implicit none 

            integer,intent(in) :: natm
            real(8),intent(in) :: alpha_free(1:natm),Reff(1:natm),Rfree(1:natm)
            real(8),intent(out) :: alpha_eff(1:natm)

            !* scratch 
            integer :: ii

            do ii=1,natm
                alpha_eff(ii) = alpha_free(ii) * (Reff(ii)/Rfree(ii))**3 
            end do
        end subroutine


        subroutine design_matrix_alternative(rcuts,ks,Zks,localpos,ultrapos,Zlocal,Zultra,Natm,&
                &ultraidx,basis,phi)
            implicit none

            integer,intent(in) :: Natm(1:2),ultraidx(1:Natm(2)),basis(1:4)
            real(8),intent(in) :: localpos(1:3,1:Natm(1)),ultrapos(1:3,1:Natm(2))
            real(8),intent(in) :: Zlocal(1:Natm(1)),Zultra(1:Natm(2))
            real(8),intent(in) :: rcuts(1:basis(1)),ks(1:basis(2)),Zks(1:basis(3))
            real(8),intent(out) :: phi(1:basis(1)*basis(2)+basis(3)*basis(4),1:Natm(1))

            !* scratch
            integer :: cntr,ii,jj,kk,ll,nn
            real(8) :: localgamma(1:Natm(1)),ultragamma(1:Natm(2)) 
            real(8) :: rtol,rcut2,dr2(1:Natm(2),1:Natm(1)),rcut,invk2,Zk

            rtol = 0.0000001d0
            rcut2 = rcut**2

            !* calculate displacement matrix
            call displacement_matrix(localpos,ultrapos,Natm,dr2)

            cntr = 0

            do ii=1,basis(1),1
                rcut = rcuts(ii)

                do jj=1,basis(2),1
                    invk2 = 1.0d0/(ks(jj)**2)

                    do kk=1,basis(3),1
                        Zk = Zks(kk)
                        
                        !* recalculate brokherde feature
                        call calculate_gamma(Zultra,Natm,ultraidx,rcut,dr2,&
                                &invk2,Zk,localgamma,ultragamma)
                   
                        cntr = (ii-1)*(jj-1)*(kk-1)*basis(4)

                        do nn=1,Natm(1),1
                            do ll=1,basis(4),1
                                phi(cntr+ll,nn) = cos(localgamma(nn)*dble(ll))
                            end do
                        end do !* end nn
                    
                    end do !* end kk
                end do !* end jj
            end do !* end ii

        end subroutine design_matrix_alternative

        subroutine design_matrix_singlestructure(rcut,localpos,ultrapos,Zlocal,Zultra,Natm,&
                &invk2,Zk,ultraidx,basis,phi)
            !===================================================================!
            ! return the design matrix segment for a single structure           !
            !-------------------------------------------------------------------!
            ! Input
            ! -----
            !   - 
            !===================================================================!
            
            use rvm_basis, only : wrapped_smooth
            
            implicit none
           
            !* arguments
            integer,intent(in) :: Natm(1:2),ultraidx(1:Natm(2)),basis(1:4)
            real(8),intent(in) :: rcut,localpos(1:3,1:Natm(1)),ultrapos(1:3,1:Natm(2))
            real(8),intent(in) :: Zlocal(1:Natm(1)),Zultra(1:Natm(2)),invk2,Zk
            real(8),intent(out) :: phi(1:basis(1)*basis(2)+basis(3)*basis(4),1:Natm(1))

            !* scratch
            integer :: ii,jj,ki,kj,cntr
            real(8) :: localgamma(1:Natm(1)),ultragamma(1:Natm(2)) 
            real(8) :: dr2(1:Natm(2),1:Natm(1)),rtol,tmp,rcut2,gamii
          
            rtol = 0.0000001d0
            rcut2 = rcut**2

            !* calculate displacement matrix
            call displacement_matrix(localpos,ultrapos,Natm,dr2)

            !* Brockherde feature
            call calculate_gamma(Zultra,Natm,ultraidx,rcut,dr2,&
                    &invk2,Zk,localgamma,ultragamma)
            
            do ii=1,Natm(1)
                cntr = 1
                
                gamii = localgamma(ii)
                
                !---------------------!
                !* f(Zi,gammai) term *!
                !---------------------!

                do ki=0,basis(1)-1
                    do kj=0,basis(2)-1
                        phi(cntr,ii) = cos(Zlocal(ii)*dble(ki))*cos(gamii*dble(kj))
                        cntr = cntr + 1
                    end do
                end do

                !----------------------------------!
                !* sum_j g(dr,gammai,gammaj) term *!
                !----------------------------------!
            
                do ki=0,basis(3)-1
                    do kj=0,basis(4)-1
    
                        phi(cntr,ii) = 0.0d0

                        do jj=1,Natm(2)
                            tmp = dr2(jj,ii)
                            
                            if ( (tmp.lt.rtol).or.(tmp.ge.rcut2) ) then
                                cycle
                            end if

                            tmp = sqrt(tmp)
                            
                            phi(cntr,ii) = phi(cntr,ii) + cos(dble(ki)*ultragamma(jj))*&
                                    &cos(dble(ki)*gamii)*cos(dble(kj)*tmp)*&
                                    &wrapped_smooth(0.2d0,rcut,.true.,tmp)
                        end do ! end jj
                    
                    cntr = cntr + 1
                    end do ! end kj
                end do ! end ki
            end do ! end ii
        end subroutine design_matrix_singlestructure

        subroutine calculate_gamma(Zultra,Natm,ultraidx,rcut,dr2,&
                &invk2,Zk,localgamma,ultragamma)
            !===================================================================!
            ! calculate gamma for local atoms and then copy to ultra atoms      !
            !===================================================================!
            implicit none

            !* arguments
            integer,intent(in) :: Natm(1:2),ultraidx(1:Natm(2))
            real(8),intent(in) :: rcut,Zultra(1:Natm(2)),dr2(1:Natm(2),1:Natm(1))
            real(8),intent(in) :: invk2,Zk
            real(8),intent(out) :: localgamma(1:Natm(1)),ultragamma(1:Natm(2))

            !* scratch
            integer :: ii
            
            do ii=1,Natm(1)
               localgamma(ii) = gamma_gaussian(dr2(:,ii),Zultra,Natm(2),invk2,Zk,rcut)
            end do

            !* distribute gamma to duplicate atoms
            do ii=1,Natm(2)
                ultragamma(ii) = localgamma(ultraidx(ii))
            end do
        end subroutine calculate_gamma
        
        subroutine calculate_normal(Zultra,Zlocal,Natm,rcut,dr2,&
                &mean,normal_precision,zi_pow_const,zj_pow_const,localgamma)
            !===================================================================!
            ! calculate gamma for local atoms and then copy to ultra atoms      !
            !===================================================================!
            implicit none

            !* arguments
            integer,intent(in) :: Natm(1:2)
            real(8),intent(in) :: rcut,Zultra(1:Natm(2)),dr2(1:Natm(2),1:Natm(1))
            real(8),intent(in) :: mean,Zlocal(1:Natm(1)),normal_precision
            real(8),intent(in) :: zi_pow_const,zj_pow_const
            real(8),intent(out) :: localgamma(1:Natm(1))

            !* scratch
            integer :: ii
            
            do ii=1,Natm(1)
                !* loop over local atoms
                localgamma(ii) = feature_normal(dr2(:,ii),Zultra,Natm(2),Zlocal(ii),rcut,&
                        &mean,normal_precision,zi_pow_const,zj_pow_const)
            end do

        end subroutine calculate_normal


        subroutine phi_multiple_structures(nfiles,files,natm,rcuts,gammaks,Zks,basis,&
                &Nbases,phi_type,phi)
            !===================================================================!
            !                                                                   !
            !                                                                   !
            ! Input                                                             !
            ! -----                                                             !
            !   - files  : array of 1024 char strings giving full file path to  !
            !              input                                                !
            !   - nfiles : number of files/structures to read                   !
            !   - natm   : integer array giving number of atoms in each         !
            !              structure                                            !
            !   - rcut   : interaction cut off radius                           !
            !   - gammak : parameter value for feature                          !
            !   - basis  : int array of length 4 giving number of basis         !
            !              functions for terms in model                         !
            !   - phi    : design matrix with (basis function,observation)      !
            !              page ordering                                        !
            !===================================================================!
            
            use io, only : readfile
            use util, only : get_ultracellf90

            implicit none

            integer,intent(in) :: nfiles,natm(1:nfiles),basis(1:4),Nbases,phi_type
            character(len=1024),dimension(nfiles),intent(in) :: files
            real(8),intent(in) :: gammaks(:),rcuts(:),Zks(:)
            real(8),intent(out) :: phi(1:Nbases,1:sum(natm))
        
        
            !* scratch
            integer :: ss,start_idx,end_idx,parse_natm(1:2)
            real(8),allocatable :: localcart(:,:),localfrac(:,:),localZ(:)
            real(8),allocatable :: ultracart(:,:),ultraZ(:)
            integer,allocatable :: ultraidx(:)
            character(len=1024) :: string
            real(8) :: cell(1:3,1:3)

            do ss=1,nfiles
                if (ss.gt.1) then
                    start_idx = sum(natm(1:ss-1)) + 1
                else
                    start_idx = 1
                end if
                end_idx = sum(natm(1:ss))
                
                allocate(localfrac(3,natm(ss)))
                allocate(localcart(3,natm(ss)))
                allocate(localZ(natm(ss)))
                
                !* passing trim(files(ss)) does not work
                string = trim(files(ss))
                
                !* read fractional coordinates,cell,atomic numbers
                call readfile(string,0,cell,natm(ss),localfrac,localZ)

                !* cartesian coordinates
                call global_frac_to_cart(localfrac,cell,size(localZ),localcart)

                !* ultra cell calcuation
                call get_ultracellf90(localfrac,localcart,cell,localZ,size(localZ),maxval(rcuts),10000,&
                        &ultracart,ultraidx,ultraZ)

                !* atoms in local and ultracell
                parse_natm(1) = natm(ss)
                parse_natm(2) = size(ultraZ)

                !*------------------------------------*!
                !* calculate segment of design matrix *!
                !*------------------------------------*!
                
                if (phi_type.eq.1) then
                    call design_matrix_singlestructure(rcuts(1),localcart,ultracart,localZ,ultraZ,&
                            &parse_natm,1.0d0/(gammaks(1)**2),Zks(1),ultraidx,basis,phi(:,start_idx:end_idx))
                else if (phi_type.eq.2) then
                    call design_matrix_alternative(rcuts,gammaks,Zks,localcart,ultracart,localZ,ultraZ,&
                            &parse_natm,ultraidx,basis,phi(:,start_idx:end_idx))
                end if

                deallocate(localcart)
                deallocate(localfrac)
                deallocate(localZ)
                deallocate(ultracart)
                deallocate(ultraZ)
                deallocate(ultraidx)

            end do  !* loop over structures

        end subroutine phi_multiple_structures
        
        
        integer function num_atom_atom_distances(nfiles,files,natm,rcut)
            !===================================================================!
            ! return the total number of atom-atom distances
            !===================================================================!
            
            use io, only : readfile
            use util, only : get_ultracellf90,atomatom_distances

            implicit none

            integer,intent(in) :: nfiles,natm(1:nfiles)
            character(len=1024),dimension(nfiles),intent(in) :: files
            real(8),intent(in) :: rcut
        
        
            !* scratch
            integer :: ss,start_idx,end_idx,parse_natm(1:2)
            real(8),allocatable :: localcart(:,:),localfrac(:,:),localZ(:)
            real(8),allocatable :: ultracart(:,:),ultraZ(:)
            integer,allocatable :: ultraidx(:)
            character(len=1024) :: string
            real(8) :: cell(1:3,1:3)
            real(8),allocatable :: tmp_disp(:),dr2(:,:)
            logical :: have_allocated_mem
            integer :: num_distances

            num_distances = 0

            do ss=1,nfiles
                if (ss.gt.1) then
                    start_idx = sum(natm(1:ss-1)) + 1
                else
                    start_idx = 1
                end if
                end_idx = sum(natm(1:ss))
                
                allocate(localfrac(3,natm(ss)))
                allocate(localcart(3,natm(ss)))
                allocate(localZ(natm(ss)))
                
                !* passing trim(files(ss)) does not work
                string = trim(files(ss))
                
                !* read fractional coordinates,cell,atomic numbers
                call readfile(string,0,cell,natm(ss),localfrac,localZ)

                !* cartesian coordinates
                call global_frac_to_cart(localfrac,cell,size(localZ),localcart)

                !* ultra cell calcuation
                call get_ultracellf90(localfrac,localcart,cell,localZ,size(localZ),rcut,10000,&
                        &ultracart,ultraidx,ultraZ)

                !* atoms in local and ultracell
                parse_natm(1) = natm(ss)
                parse_natm(2) = size(ultraZ)

                allocate(dr2(parse_natm(2),parse_natm(1)))

                !* calculate displacement matrix
                call displacement_matrix(localcart,ultracart,parse_natm,dr2)

                !* get array of atom-atom distances
                call atomatom_distances(parse_natm,dr2,rcut,tmp_disp,have_allocated_mem)


                if (have_allocated_mem) then
                    !* check THIS 
                    num_distances = num_distances + size(tmp_disp)
                    deallocate(tmp_disp)
                end if

                deallocate(localcart)
                deallocate(localfrac)
                deallocate(localZ)
                deallocate(ultracart)
                deallocate(ultraZ)
                deallocate(ultraidx)
                deallocate(dr2)
            end do  !* loop over structures

            num_atom_atom_distances = num_distances
        end function num_atom_atom_distances

        subroutine atom_atom_distances(nfiles,files,natm,rcut,num_dist,distances)
            !===================================================================!
            ! store atom-atom distance in distances
            !===================================================================!
            
            use io, only : readfile
            use util, only : get_ultracellf90,atomatom_distances

            implicit none

            integer,intent(in) :: nfiles,natm(1:nfiles),num_dist
            character(len=1024),dimension(nfiles),intent(in) :: files
            real(8),intent(in) :: rcut
            real(8),intent(out) :: distances(1:num_dist)
        
        
            !* scratch
            integer :: ss,start_idx,end_idx,parse_natm(1:2)
            real(8),allocatable :: localcart(:,:),localfrac(:,:),localZ(:)
            real(8),allocatable :: ultracart(:,:),ultraZ(:)
            integer,allocatable :: ultraidx(:)
            character(len=1024) :: string
            real(8) :: cell(1:3,1:3)
            real(8),allocatable :: tmp_disp(:),dr2(:,:)
            logical :: have_allocated_mem
            integer :: offset

            offset = 1

            do ss=1,nfiles
                if (ss.gt.1) then
                    start_idx = sum(natm(1:ss-1)) + 1
                else
                    start_idx = 1
                end if
                end_idx = sum(natm(1:ss))
                
                allocate(localfrac(3,natm(ss)))
                allocate(localcart(3,natm(ss)))
                allocate(localZ(natm(ss)))
                
                !* passing trim(files(ss)) does not work
                string = trim(files(ss))
                
                !* read fractional coordinates,cell,atomic numbers
                call readfile(string,0,cell,natm(ss),localfrac,localZ)

                !* cartesian coordinates
                call global_frac_to_cart(localfrac,cell,size(localZ),localcart)

                !* ultra cell calcuation
                call get_ultracellf90(localfrac,localcart,cell,localZ,size(localZ),rcut,10000,&
                        &ultracart,ultraidx,ultraZ)

                !* atoms in local and ultracell
                parse_natm(1) = natm(ss)
                parse_natm(2) = size(ultraZ)

                allocate(dr2(parse_natm(2),parse_natm(1)))

                !* calculate displacement matrix
                call displacement_matrix(localcart,ultracart,parse_natm,dr2)

                !* get array of atom-atom distances
                call atomatom_distances(parse_natm,dr2,rcut,tmp_disp,have_allocated_mem)


                if (have_allocated_mem) then
                    distances(offset:offset+size(tmp_disp)-1) = tmp_disp(:)
                       
                    !* check THIS 
                    offset = offset + size(tmp_disp)
                    deallocate(tmp_disp)
                end if

                deallocate(localcart)
                deallocate(localfrac)
                deallocate(localZ)
                deallocate(ultracart)
                deallocate(ultraZ)
                deallocate(ultraidx)
                deallocate(dr2)
            end do  !* loop over structures

        end subroutine atom_atom_distances
        
        integer function threebody_angular_info(nfiles,files,natm,rcut,query,nangles,angles_all)
            !===================================================================!
            ! store atom-atom distance in distances
            !===================================================================!
            
            use io, only : readfile
            use util, only : get_ultracellf90,atomatom_distances,angular_info_structure

            implicit none

            integer,intent(in) :: nfiles,natm(1:nfiles),nangles
            character(len=1024),dimension(nfiles),intent(in) :: files
            real(8),intent(in) :: rcut
            logical,intent(in) :: query
            real(8),intent(out) :: angles_all(1:3,1:nangles)
        
        
            !* scratch
            integer :: ss,start_idx,end_idx,parse_natm(1:2)
            real(8),allocatable :: localcart(:,:),localfrac(:,:),localZ(:)
            real(8),allocatable :: ultracart(:,:),ultraZ(:)
            integer,allocatable :: ultraidx(:)
            character(len=1024) :: string
            real(8) :: cell(1:3,1:3)
            real(8),allocatable :: angles_single(:,:)
            integer :: offset,buffer_size,cntr
            
            offset = 1

            if (query) then
                cntr = 0
            else
                cntr = 1
            end if

            buffer_size = 1000

            do ss=1,nfiles
                if (ss.gt.1) then
                    start_idx = sum(natm(1:ss-1)) + 1
                else
                    start_idx = 1
                end if
                end_idx = sum(natm(1:ss))
                
                allocate(localfrac(3,natm(ss)))
                allocate(localcart(3,natm(ss)))
                allocate(localZ(natm(ss)))
                
                !* passing trim(files(ss)) does not work
                string = trim(files(ss))
                
                !* read fractional coordinates,cell,atomic numbers
                call readfile(string,0,cell,natm(ss),localfrac,localZ)

                !* cartesian coordinates
                call global_frac_to_cart(localfrac,cell,size(localZ),localcart)

                !* ultra cell calcuation
                call get_ultracellf90(localfrac,localcart,cell,localZ,size(localZ),rcut,10000,&
                        &ultracart,ultraidx,ultraZ)

                !* atoms in local and ultracell
                parse_natm(1) = natm(ss)
                parse_natm(2) = size(ultraZ)

                !* fetch list of angular information
                call angular_info_structure(localcart,ultracart,localZ,ultraZ,&
                        &parse_natm,rcut,buffer_size,angles_single)
                
                if (query.neqv..true.) then
                    angles_all(:,cntr:cntr+int(float(size(angles_single))/7.0)-1) = angles_single(1:3,:)        
                end if
                
                cntr = cntr + int(float(size(angles_single))/7.0)
                

                deallocate(localcart)
                deallocate(localfrac)
                deallocate(localZ)
                deallocate(ultracart)
                deallocate(ultraZ)
                deallocate(ultraidx)
                deallocate(angles_single)
            end do  !* loop over structures
            
            if (query) then
                threebody_angular_info = cntr
            else
                threebody_angular_info = -1
            end if
        end function
        
        subroutine phi_multiple_structures_openmp(nfiles,files,natm,rcuts,gammaks,Zks,basis,&
                &Nbases,phi_type,phi)
            !===================================================================!
            !                                                                   !
            !                                                                   !
            ! Input                                                             !
            ! -----                                                             !
            !   - files  : array of 1024 char strings giving full file path to  !
            !              input                                                !
            !   - nfiles : number of files/structures to read                   !
            !   - natm   : integer array giving number of atoms in each         !
            !              structure                                            !
            !   - rcut   : interaction cut off radius                           !
            !   - gammak : parameter value for feature                          !
            !   - basis  : int array of length 4 giving number of basis         !
            !              functions for terms in model                         !
            !   - phi    : design matrix with (basis function,observation)      !
            !              page ordering                                        !
            !===================================================================!
            
            !$ use omp_lib
            use io, only : readfile
            use util, only : get_ultracellf90

            implicit none

            integer,intent(in) :: nfiles,natm(1:nfiles),basis(1:4),Nbases,phi_type
            character(len=1024),dimension(nfiles),intent(in) :: files
            real(8),intent(in) :: gammaks(:),rcuts(:),Zks(:)
            real(8),intent(out) :: phi(1:Nbases,1:sum(natm))
        
        
            !* scratch
            integer :: ss,start_idx,end_idx,parse_natm(1:2),atoms_start(1:nfiles)
            integer :: atoms_end(1:nfiles)
            real(8),allocatable :: localcart(:,:)
            real(8),allocatable :: ultracart(:,:),ultraZ(:)
            integer,allocatable :: ultraidx(:)
            character(len=1024) :: string
            real(8) :: cellall(1:3,1:3,1:sum(natm)),invk2,localfracall(1:3,1:sum(natm))
            real(8) :: localZall(1:sum(natm))

            !* openmp
            integer :: thread,num_threads,ds,thread_start,thread_end


            !* cannot garauntee read works in parallel
            do ss=1,nfiles
                if (ss.ne.1) then
                    atoms_start(ss) = sum(natm(1:ss-1)) + 1
                else
                    atoms_start(ss) = 1
                end if
                atoms_end(ss) = sum(natm(1:ss))
                
                !* passing trim(files(ss)) does not work
                string = trim(files(ss))

                call readfile(string,0,cellall(:,:,ss),natm(ss),&
                        &localfracall(1:3,atoms_start(ss):atoms_end(ss)),&
                        &localZall(atoms_start(ss):atoms_end(ss)))
            end do
            
            !$omp parallel num_threads(omp_get_max_threads()),&
            !$omp& private(start_idx,end_idx,localcart,ultracart,ultraZ),&
            !$omp& private(ss,parse_natm,ultraidx,thread,num_threads,ds,thread_start,thread_end),&
            !$omp& shared(phi,natm,files,nfiles,basis,gammaks,rcuts,invk2,localfracall,localZall),&
            !$omp& shared(atoms_start,atoms_end,string,cellall,Zks)

            !* thread idx = [0,num_threads-1] ?
            thread = omp_get_thread_num()

            !* total number of threads
            num_threads = omp_get_max_threads()

            !* number of files per thread except for final thread
            ds = int(floor(float(nfiles)/float(num_threads)))

            !* always true
            thread_start = thread*ds + 1

            if (thread.eq.num_threads-1) then
                thread_end = nfiles
            else
                thread_end = (thread+1)*ds
            end if

            do ss=thread_start,thread_end
                if (ss.gt.1) then
                    start_idx = sum(natm(1:ss-1)) + 1
                else
                    start_idx = 1
                end if
                end_idx = sum(natm(1:ss))
                
                allocate(localcart(3,natm(ss)))
                
                !* cartesian coordinates
                call global_frac_to_cart(localfracall(1:3,atoms_start(ss):atoms_end(ss)),cellall(:,:,ss),&
                        &natm(ss),localcart)
                
                !* ultra cell calcuation
                call get_ultracellf90(localfracall(1:3,atoms_start(ss):atoms_end(ss)),localcart,cellall(:,:,ss),&
                        &localZall(atoms_start(ss):atoms_end(ss)),natm(ss),maxval(rcuts),10000,&
                        &ultracart,ultraidx,ultraZ)

                !* atoms in local and ultracell
                parse_natm(1) = natm(ss)
                parse_natm(2) = size(ultraZ)
               
                if (phi_type.eq.1) then
                    !* calculate segment of design matrix
                    call design_matrix_singlestructure(rcuts(1),localcart,ultracart,&
                            &localZall(atoms_start(ss):atoms_end(ss)),ultraZ,&
                            &parse_natm,1.0d0/(gammaks(1)**2),Zks(1),ultraidx,basis,phi(:,start_idx:end_idx))
                else if (phi_type.eq.2) then
                    call design_matrix_alternative(rcuts,gammaks,Zks,localcart,ultracart,&
                            &localZall(atoms_start(ss):atoms_end(ss)),ultraZ,parse_natm,&
                            &ultraidx,basis,phi(:,start_idx:end_idx))
                end if

                deallocate(localcart)
                deallocate(ultracart)
                deallocate(ultraZ)
                deallocate(ultraidx)
            
            end do  !* loop over structures
            
            !$omp end parallel

        end subroutine phi_multiple_structures_openmp


        subroutine global_frac_to_cart(frac,cell,natm,cart)
            implicit none

            integer,intent(in) :: natm
            real(8),intent(in) :: frac(1:3,1:natm),cell(1:3,1:3)
            real(8),intent(out) :: cart(1:3,1:natm)

            !* scratch
            integer :: ii,jj,kk

            do ii=1,natm
                cart(1:3,ii) = 0.0d0
                do jj=1,3
                    do kk=1,3
                        cart(jj,ii) = cart(jj,ii) + cell(jj,kk)*frac(kk,ii)
                    end do
                end do
            end do
        end subroutine
        
        subroutine gamma_multiple_structures_openmp(nfiles,files,natm,rcut,parameters,num_parameters,&
                &gamma_out)
            !===================================================================!
            !                                                                   !
            !                                                                   !
            ! Input                                                             !
            ! -----                                                             !
            !   - files  : array of 1024 char strings giving full file path to  !
            !              input                                                !
            !   - nfiles : number of files/structures to read                   !
            !   - natm   : integer array giving number of atoms in each         !
            !              structure                                            !
            !   - rcut   : interaction cut off radius                           !
            !   - gammak : parameter value for feature                          !
            !   - basis  : int array of length 4 giving number of basis         !
            !              functions for terms in model                         !
            !   - phi    : design matrix with (basis function,observation)      !
            !              page ordering                                        !
            !===================================================================!
            
            !$ use omp_lib
            use io, only : readfile
            use util, only : get_ultracellf90

            implicit none

            integer,intent(in) :: nfiles,natm(1:nfiles),num_parameters
            character(len=1024),dimension(nfiles),intent(in) :: files
            real(8),intent(in) :: rcut,parameters(1:num_parameters)
            real(8),intent(out) :: gamma_out(1:sum(natm))
        
        
            !* scratch
            integer :: ss,start_idx,end_idx,parse_natm(1:2),atoms_start(1:nfiles)
            integer :: atoms_end(1:nfiles),feature_type
            real(8),allocatable :: localcart(:,:)
            real(8),allocatable :: ultracart(:,:),ultraZ(:)
            real(8),allocatable :: dr2(:,:)
            real(8),allocatable :: localgamma(:),ultragamma(:) 
            integer,allocatable :: ultraidx(:)
            character(len=1024) :: string
            real(8) :: cellall(1:3,1:3,1:sum(natm)),localfracall(1:3,1:sum(natm))
            real(8) :: localZall(1:sum(natm)),invk2
            real(8) :: rtol,Zk,mean,normal_precision,zi_pow_const,zj_pow_const

            !* openmp
            integer :: thread,num_threads,ds,thread_start,thread_end


            if (num_parameters.eq.2) then
                feature_type = 1
            else if (num_parameters.eq.4) then
                feature_type = 2
            else
                write(*,*) 'Unrecognised feature type'
                call exit(0)
            end if

            !* parameters(1) = gammak
            !* parameters(2) = Zk


            !* cannot garauntee read works in parallel
            do ss=1,nfiles
                if (ss.ne.1) then
                    atoms_start(ss) = sum(natm(1:ss-1)) + 1
                else
                    atoms_start(ss) = 1
                end if
                atoms_end(ss) = sum(natm(1:ss))
                
                !* passing trim(files(ss)) does not work
                string = trim(files(ss))

                call readfile(string,0,cellall(:,:,ss),natm(ss),&
                        &localfracall(1:3,atoms_start(ss):atoms_end(ss)),&
                        &localZall(atoms_start(ss):atoms_end(ss)))
            end do
            
            !$omp parallel num_threads(omp_get_max_threads()),&
            !$omp& private(start_idx,end_idx,localcart,ultracart,ultraZ),&
            !$omp& private(ss,parse_natm,ultraidx,thread,num_threads,ds,thread_start,thread_end),&
            !$omp& private(dr2,ultragamma,localgamma),&
            !$omp& shared(natm,files,nfiles,rcut,localfracall,localZall),&
            !$omp& shared(atoms_start,atoms_end,string,cellall,Zk,rtol,mean,normal_precision)

            !* thread idx = [0,num_threads-1] ?
            thread = omp_get_thread_num()

            !* total number of threads
            num_threads = omp_get_max_threads()

            !* number of files per thread except for final thread
            ds = int(floor(float(nfiles)/float(num_threads)))

            !* always true
            thread_start = thread*ds + 1

            if (thread.eq.num_threads-1) then
                thread_end = nfiles
            else
                thread_end = (thread+1)*ds
            end if

            do ss=thread_start,thread_end
                if (ss.gt.1) then
                    start_idx = sum(natm(1:ss-1)) + 1
                else
                    start_idx = 1
                end if
                end_idx = sum(natm(1:ss))
                
                allocate(localcart(3,natm(ss)))
                
                !* cartesian coordinates
                call global_frac_to_cart(localfracall(1:3,atoms_start(ss):atoms_end(ss)),&
                        &cellall(:,:,ss),natm(ss),localcart)
                
                !* ultra cell calcuation
                call get_ultracellf90(localfracall(1:3,atoms_start(ss):atoms_end(ss)),localcart,&
                        &cellall(:,:,ss),localZall(atoms_start(ss):atoms_end(ss)),natm(ss),&
                        &rcut,10000,ultracart,ultraidx,ultraZ)

                !* atoms in local and ultracell
                parse_natm(1) = natm(ss)
                parse_natm(2) = size(ultraZ)
                
                rtol = 0.0000001d0


                allocate(dr2(parse_natm(2),parse_natm(1)))
                allocate(localgamma(parse_natm(1)))
                allocate(ultragamma(parse_natm(2)))

                !* calculate displacement matrix
                call displacement_matrix(localcart,ultracart,parse_natm,dr2)

                if (feature_type.eq.1) then
                    !* Brockherde feature
                    invk2 = 1.0d0/(parameters(1)**2)
                    Zk = parameters(2)

                    call calculate_gamma(ultraZ,parse_natm,ultraidx,rcut,dr2,&
                            &invk2,Zk,localgamma,ultragamma)
                   
                    gamma_out(start_idx:end_idx) = localgamma(:)
                else if (feature_type.eq.2) then
                    !* normal feature
    
                    mean = parameters(1)
                    normal_precision = parameters(2)

                    !* power for atomic numbers
                    zi_pow_const = parameters(3)
                    zj_pow_const = parameters(4)

                    call calculate_normal(ultraZ,localZall(atoms_start(ss):atoms_end(ss)),&
                            &parse_natm,rcut,dr2,mean,normal_precision,&
                            &zi_pow_const,zj_pow_const,localgamma)
                    
                    gamma_out(start_idx:end_idx) = localgamma(:)
                end if
                
                deallocate(localcart)
                deallocate(ultracart)
                deallocate(ultraZ)
                deallocate(ultraidx)
                deallocate(dr2)
                deallocate(localgamma)
                deallocate(ultragamma)
            end do  !* loop over structures
            
            !$omp end parallel

        end subroutine gamma_multiple_structures_openmp
        
        subroutine anisotropic_structure_features(localpos,ultrapos,localz,&
                &ultraz,natm,rcut,num_components,means,precisions,&
                &zi_pow_const,zj_pow_const,feature_out)
            use util, only : angular_info

            implicit none

            integer,intent(in) :: natm(1:2),num_components
            real(8),intent(in) :: localpos(1:natm(1)),ultrapos(1:natm(2))
            real(8),intent(in) :: rcut,localz(1:natm(1)),ultraz(1:natm(2))
            real(8),intent(in) :: means(1:3,1:num_components)
            real(8),intent(in) :: precisions(1:3,1:3,1:num_components)
            real(8),intent(in) :: zi_pow_const,zj_pow_const
            real(8),intent(out) :: feature_out(1:num_components,1:natm(1))

            !* scratch
            real(8),allocatable :: aniso_info(:,:)
            integer :: atom,num_bonds
            
            do atom=1,natm(1),1

                if (angular_info(atom,localpos,ultrapos,localz,ultraz,natm,&
                        &rcut,aniso_info) ) then
                    !* atom has neighbours
                    
                    num_bonds = int(float(size(aniso_info))/7.0)
                    call anisotropic_feature(aniso_info,num_components,&
                            &num_bonds,means,precisions,zi_pow_const,zj_pow_const,&
                            &rcut,feature_out(:,atom))
                    deallocate(aniso_info)
                else
                    !* atom does not have neighbours
                    feature_out(:,atom) = 0.0d0
                end if
            end do

        end subroutine anisotropic_structure_features
       
        subroutine anisotropic_feature(aniso_info,num_components,num_bonds,means,&
                &precisions,zi_pow_const,zj_pow_const,rcut,feature_out)
            use rvm_basis, only : wrapped_smooth
            use util, only : multiple_determinant
            
            implicit none

            integer,intent(in) :: num_components,num_bonds
            real(8),intent(in) :: aniso_info(1:7,1:num_bonds),rcut
            real(8),intent(in) :: means(1:3,1:num_components)
            real(8),intent(in) :: precisions(1:3,1:3,1:num_components)
            real(8),intent(in) :: zi_pow_const,zj_pow_const
            real(8),intent(out) :: feature_out(1:num_components)

            !* scratch
            integer :: bond,ii,jj,kk
            real(8) :: tmp_atomic_num,tmp_taper,tmp_exp
            real(8) :: norm_const(1:num_components)


            !* | Lambda | 
            call multiple_determinant(num_components,precisions,norm_const)
            
            !* |Lambda|^0.5 / (2 pi)^{d/2}
            norm_const = sqrt(norm_const)/( 6.28318530718d0**(1.5d0) )

            feature_out(:) = 0.0d0

            do bond=1,num_bonds,1
                !* atomic number product
                tmp_atomic_num = ((aniso_info(4,bond)+1.0d0)**zi_pow_const) * &
                        &((aniso_info(5,bond)+1.0d0)**zj_pow_const) * &
                        &((aniso_info(6,bond)+1.0d0)**zj_pow_const)

                !* tapering function product
                tmp_taper = wrapped_smooth(0.2d0,rcut,.true.,aniso_info(2,bond))*& 
                        &wrapped_smooth(0.2d0,rcut,.true.,aniso_info(3,bond))*& 
                        &wrapped_smooth(0.2d0,rcut,.true.,aniso_info(7,bond)) 

                do kk=1,num_components,1
                    !* exponent
                    tmp_exp = 0.0d0
                    do ii=1,3,1
                        do jj=1,3,1
                            tmp_exp = tmp_exp + (aniso_info(ii,bond)-means(ii,kk))*&
                                    &precisions(ii,jj,kk)*(aniso_info(jj,bond)-means(jj,kk))
                        end do
                    end do
                    
                    feature_out(kk) = feature_out(kk) + norm_const(kk)*tmp_taper*tmp_atomic_num*&
                            &exp(-0.5*tmp_exp)
                end do
            end do
        end subroutine anisotropic_feature

        subroutine anisotropic_feature_multiple_structures_openmp(nfiles,files,natm,rcut,num_components,&
                &means,precisions,zi_pow_const,zj_pow_const,feature_out)
            !===================================================================!
            !                                                                   !
            !                                                                   !
            ! Input                                                             !
            ! -----                                                             !
            !   - files  : array of 1024 char strings giving full file path to  !
            !              input                                                !
            !   - nfiles : number of files/structures to read                   !
            !   - natm   : integer array giving number of atoms in each         !
            !              structure                                            !
            !   - rcut   : interaction cut off radius                           !
            !   - gammak : parameter value for feature                          !
            !   - basis  : int array of length 4 giving number of basis         !
            !              functions for terms in model                         !
            !   - phi    : design matrix with (basis function,observation)      !
            !              page ordering                                        !
            !===================================================================!
            
            !$ use omp_lib
            use io, only : readfile
            use util, only : get_ultracellf90

            implicit none

            integer,intent(in) :: nfiles,natm(1:nfiles),num_components
            character(len=1024),dimension(nfiles),intent(in) :: files
            real(8),intent(in) :: rcut,means(1:3,1:num_components)
            real(8),intent(in) :: precisions(1:3,1:3,1:num_components)
            real(8),intent(in) :: zi_pow_const,zj_pow_const
            real(8),intent(out) :: feature_out(1:num_components,1:sum(natm))
        
        
            !* scratch
            integer :: ss,start_idx,end_idx,parse_natm(1:2),atoms_start(1:nfiles)
            integer :: atoms_end(1:nfiles)
            real(8),allocatable :: localcart(:,:)
            real(8),allocatable :: ultracart(:,:),ultraZ(:)
            real(8),allocatable :: dr2(:,:)
            real(8),allocatable :: localgamma(:),ultragamma(:) 
            integer,allocatable :: ultraidx(:)
            character(len=1024) :: string
            real(8) :: cellall(1:3,1:3,1:sum(natm)),localfracall(1:3,1:sum(natm))
            real(8) :: localZall(1:sum(natm))
            real(8) :: rtol,Zk,mean,normal_precision
        

            !* openmp
            integer :: thread,num_threads,ds,thread_start,thread_end


            !* parameters(1) = gammak
            !* parameters(2) = Zk


            !* cannot garauntee read works in parallel
            do ss=1,nfiles
                if (ss.ne.1) then
                    atoms_start(ss) = sum(natm(1:ss-1)) + 1
                else
                    atoms_start(ss) = 1
                end if
                atoms_end(ss) = sum(natm(1:ss))
                
                !* passing trim(files(ss)) does not work
                string = trim(files(ss))

                call readfile(string,0,cellall(:,:,ss),natm(ss),&
                        &localfracall(1:3,atoms_start(ss):atoms_end(ss)),&
                        &localZall(atoms_start(ss):atoms_end(ss)))
            end do
            
            !$omp parallel num_threads(omp_get_max_threads()),&
            !$omp& private(start_idx,end_idx,localcart,ultracart,ultraZ),&
            !$omp& private(ss,parse_natm,ultraidx,thread,num_threads,ds,thread_start,thread_end),&
            !$omp& private(dr2,ultragamma,localgamma),&
            !$omp& shared(natm,files,nfiles,rcut,localfracall,localZall),&
            !$omp& shared(atoms_start,atoms_end,string,cellall,Zk,rtol,mean,normal_precision)

            !* thread idx = [0,num_threads-1] ?
            thread = omp_get_thread_num()

            !* total number of threads
            num_threads = omp_get_max_threads()

            !* number of files per thread except for final thread
            ds = int(floor(float(nfiles)/float(num_threads)))

            !* always true
            thread_start = thread*ds + 1

            if (thread.eq.num_threads-1) then
                thread_end = nfiles
            else
                thread_end = (thread+1)*ds
            end if

            do ss=thread_start,thread_end
                if (ss.gt.1) then
                    start_idx = sum(natm(1:ss-1)) + 1
                else
                    start_idx = 1
                end if
                end_idx = sum(natm(1:ss))
                
                allocate(localcart(3,natm(ss)))
                
                !* cartesian coordinates
                call global_frac_to_cart(localfracall(1:3,atoms_start(ss):atoms_end(ss)),&
                        &cellall(:,:,ss),natm(ss),localcart)
                
                !* ultra cell calcuation
                call get_ultracellf90(localfracall(1:3,atoms_start(ss):atoms_end(ss)),localcart,&
                        &cellall(:,:,ss),localZall(atoms_start(ss):atoms_end(ss)),natm(ss),&
                        &rcut,10000,ultracart,ultraidx,ultraZ)

                !* atoms in local and ultracell
                parse_natm(1) = natm(ss)
                parse_natm(2) = size(ultraZ)

                call anisotropic_structure_features(localcart,ultracart,&
                        &localZall(atoms_start(ss):atoms_end(ss)),&
                        &ultraz,parse_natm,rcut,num_components,means,precisions,&
                        &zi_pow_const,zj_pow_const,feature_out(:,start_idx:end_idx))
                

                
                deallocate(localcart)
                deallocate(ultracart)
                deallocate(ultraZ)
                deallocate(ultraidx)
            end do  !* loop over structures
            
            !$omp end parallel

        end subroutine anisotropic_feature_multiple_structures_openmp

end module sedc
