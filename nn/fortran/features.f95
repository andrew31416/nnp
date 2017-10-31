module features
    use config
    use feature_config
    use feature_util
    use tapering, only : taper_1,taper_deriv_1

    implicit none

    contains
        subroutine calculate_features()
            implicit none

            integer :: set_type,conf
            real(8),allocatable :: ultra_cart(:,:)
            real(8),allocatable :: ultra_z(:)
            integer,allocatable :: ultra_idx(:)
            real(8) :: mxrcut


            !* max cut off of all interactions
            mxrcut = maxrcut(0)
            
            do set_type=1,2
                do conf=1,data_sets(set_type)%nconf
                    call get_ultracell(mxrcut,1000,set_type,conf,&
                            &ultra_cart,ultra_idx,ultra_z)

                    call calculate_isotropic_info(set_type,conf,ultra_cart,ultra_z,ultra_idx)

                    !* calculate features and their derivatives
                    call calculate_all_features(set_type,conf)

                    deallocate(ultra_z)
                    deallocate(ultra_idx)
                    deallocate(ultra_cart)
                    deallocate(feature_isotropic)
                end do
            end do

            do set_type=1,2
                do conf=1,data_sets(set_type)%nconf
                    call random_number(data_sets(set_type)%configs(conf)%x(2:,:))

                    data_sets(set_type)%configs(conf)%x(1,:) = 1.0d0
                end do
            end do
            
        end subroutine calculate_features

        subroutine calculate_all_features(set_type,conf)
            implicit none

            integer,intent(in) :: set_type,conf,ultra_idx(:)
            real(8),intent(in) :: ultra_z(:)
        
            !* scratch
            integer :: atm,ft

            do atm=1,data_sets(set_type)%configs(conf)%n
                do ft=1,feature_params%num_features
                    call evaluate_feature(ft,set_type,conf,atm)
                end do !* end loop features
            end do !* end loop atoms
        end subroutine calculate_all_features

        subroutine evaluate_feature(idx,set_type,conf,atm)
            implicit none

            integer,intent(in) :: idx,set_type,conf,atm

            integer :: ftype
            logical :: twobody,threebody

            !* type
            ftype = feature_params%info(idx)%ftype

            twobody = .false.
            threebody = .false.
            if ( (ftype.eq.1).or.(ftype.eq.3) ) then
                twobody = .true.
            else if ( (ftype.eq.2).or.(ftype.eq.4) ) then
                threebody = .true.
            end if

            if (ftype.eq.0) then
                call feature_atomicnumber_1(set_type,conf,atm,idx)
            else if (twobody) then
                call feature_twobody(set_type,conf,atm,idx)
            else if (threebody) then
                call feature_threebody()
            else 
                call error("evaluate_feature","unsupported feature type")
            end if
        end subroutine evaluate_feature

        subroutine feature_atomicnumber_1(set_type,conf,atm,idx)
            implicit none

            integer,intent(in) :: set_type,conf,atm,idx

            !* scratch
            integer :: arr_idx 

            !* weight bias takes 1st element of feature vector
            arr_idx = idx + 1

            data_sets(set_type)%configs(conf)%x(arr_idx,atm) = data_sets(set_type)%configs(conf)%z(atm)

            !* feature remains invariant to position of any atom
            data_sets(set_type)%configs(conf)%x_deriv(idx,atm)%n = 0
        end subroutine

        subroutine feature_twobody(set_type,conf,atm,ft_idx)
            implicit none
            
            integer,intent(in) :: set_type,conf,atm,ft_idx

            !* scratch
            integer :: arr_idx ,ii,jj,cntr,arg
            integer :: contrib_atms(1:data_sets(set_type)%configs(conf)%n)
            integer :: idx_to_contrib(1:data_sets(set_type)%configs(conf)%n)
            logical :: zero_neighbours 
            real(8) :: rcut

            zero_neighbours = .true.

            !* weight bias takes 1st element of feature vector
            arr_idx = ft_idx + 1

            !* interaction cut off
            rcut = feature_params%info(ft_idx)%rcut

            if (feature_isotropic(atm)%n.gt.0) then
                do ii=1,feature_isotropic(atm)%n,1
                    !* search for neighbour with cut off radius
                    if (feature_isotropic(atm)%dr(ii).le.rcut) then
                        zero_neighbours = .false.
                    end if
                end do
            end if

            if (zero_neighbours) then
                !* Null contribution
                data_sets(set_type)%configs(conf)%x(arr_idx,atm) = 0.0d0
                data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%n = 0
                return
            end if

            !* idx of central atom
            contrib_atms(1) = atm  

            cntr = 1

            do ii=1,feature_isotropic(atm)%n,1
                if ( int_in_intarray(feature_isotropic(atm)%idx(ii),contrib_atms(1:cntr),arg) ) then
                    !* Local atom already in list, note corresponding idx in contrib_atms
                    idx_to_contrib(ii) = arg
                    cycle
                else if (feature_isotropic(atm)%dr(ii).le.rcut) then
                    cntr = cntr + 1
                    !* note this local atom contributes to this feature for atom
                    contrib_atms(cntr) = feature_isotropic(atm)%idx(ii)
                    idx_to_contrib(ii) = cntr
                else
                    !* atom is beyond interaction cut off
                    idx_to_contrib(ii) = -1     ! NULL value
                end if
            end do !* end loop over neighbour images

            allocate(data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%idx(cntr))
            allocate(data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(3,cntr))
            
            !* number of atoms in local cell contributing to feature (including central atom)
            data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%n = cntr
            
            !* local indices of atoms contributing to feature
            data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%idx(:) = contrib_atms(1:cntr)

            !* zero features
            data_sets(set_type)%configs(conf)%x(arr_idx,atm) = 0.0d0
            data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(:,:) = 0.0d0

    
            !* zero^th derivative    
            do ii=1,feature_isotropic(atm)%n
                if (feature_isotropic(atm)%dr(ii).le.rcut) then
                    !* contributing interaction
                    call feature_behler_iso(atm,ii,ft_idx,data_sets(set_type)%configs(conf)%x(arr_idx,atm))

                    call feature_behler_iso_deriv(atm,ii,&
                            &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(1:3,idx_to_contrib(ii)))
                end if
            end do
            !* derivative wrt. central atm
            call feature_behler_iso_deriv(atm,0,&
                    &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(1:3,1))


        end subroutine feature_twobody

        subroutine feature_behler_iso(atm,neigh_idx,ft_idx,current_val)
            implicit none

            integer,intent(in) :: atm,neigh_idx,ft_idx
            real(8),intent(inout) :: current_val

            !* scratch
            real(8) :: dr,tmp1,tmp2,tmp3,za,zb,rcut,eta,rs
            
            dr  = feature_isotropic(atm)%dr(neigh_idx)
            
            !* symmetry function params
            za   = feature_params%info(ft_idx)%za
            zb   = feature_params%info(ft_idx)%zb
            rs   = feature_params%info(ft_idx)%rs
            fs   = feature_params%info(ft_idx)%fs
            eta  = feature_params%info(ft_idx)%eta
            rcut = feature_params%info(ft_idx)%rcut

            !* exponential
            tmp1 = exp(-eta*(dr-rs)**2)

            !* tapering
            tmp2 = taper_1(dr,rcut,fs)

            !* atomic numbers
            tmp3 = (data_sets(set_type)%configs(conf)%z(atm)+1.0d0)**za * &
                    &(feature_isotropic(atm)%z(neigh_idx)+1.0d0)**zb

            current_val = current_val + tmp1*tmp2*tmp3
        end subroutine feature_behler_iso
      
        subroutine feature_behler_iso_deriv(atm,neigh_idx,deriv_vec)
            implicit none

            integer,intent(in) :: atm,neigh_idx
            real(8),intent(inout) :: deriv_vec(1:3)

            !* scratch
            real(8) :: dr_scl,dr_vec(1:3),tap_deriv,tap,tmp1,tmp2
            real(8) :: rs,fs,rcut
            real(8) :: za,zb,eta
            integer :: ii,lim1,lim2

            !* symmetry function params
            za   = feature_params%info(ft_idx)%za
            zb   = feature_params%info(ft_idx)%zb
            rs   = feature_params%info(ft_idx)%rs
            fs   = feature_params%info(ft_idx)%fs
            eta  = feature_params%info(ft_idx)%eta
            rcut = feature_params%info(ft_idx)%rcut

            if (neigh_idx.eq.0) then
                lim1 = 1
                lim2 = feature_isotropic(atm)%n
                tmp2 = -1.0d0       !* sign for drij/d r_central
            else
                lim1 = neigh_idx
                lim2 = neigh_idx    
                tmp2 = 1.0d0        !* sign for drij/d r_neighbour
            end if


            !* derivative wrt. central atom itself
            do ii=lim1,lim2,1
                !* atom-atom distance
                dr_scl = feature_isotropic(atm)%dr(ii)

                !* r_neighbour - r_centralatom
                dr_vec(:) = feature_isotropic%drdri(:,ii)

                !* tapering
                tap = taper_1(dr,rcut,fs)
                tap_deriv = taper_deriv_1(dr,rcut,fs)

                tmp1 =  exp(-eta*(dr_scl-rs)**2)  *  (tap_deriv + &
                        &2.0d0*(eta**2)*(dr_scl-rs)*tap) 

                deriv_vec(:) = deriv_vec(:) + dr_vec(:)*tmp1*tmp2
            end do
        end subroutine feature_behler_iso_deriv

end module
