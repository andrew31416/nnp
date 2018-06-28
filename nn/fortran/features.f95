module features
    use config
    use feature_config
    use feature_util
    use tapering, only : taper_1,taper_deriv_1
    use init, only : init_set_neigh_info
    use util, only : load_balance_alg_1
   
    implicit none

    external :: dsymv

    contains
        subroutine calculate_features(scale_features,parallel,updating_features)
            implicit none

            logical,intent(in) :: parallel,scale_features,updating_features

            integer :: set_type
            
            do set_type=1,2
                !* calculate forces and stress tensor
                call calculate_features_singleset(set_type,.true.,.true.,scale_features,parallel,&
                        &updating_features)
            end do
        end subroutine calculate_features
        
        subroutine calculate_features_singleset(set_type,need_forces,need_stress,&
        &scale_features,parallel,updating_features)
            use omp_lib
            
            implicit none

            integer,intent(in) :: set_type
            logical,intent(in) :: need_forces,need_stress
            logical,intent(in) :: scale_features,parallel,updating_features
            
            real(8),allocatable :: ultra_cart(:,:)
            real(8),allocatable :: ultra_z(:)
            integer,allocatable :: ultra_idx(:)
            real(8) :: mxrcut,buffersizeMB
            logical :: calc_twobody,calc_threebody
            integer :: conf,buffersizeDBLS

            !* openMP variables
            integer :: thread_idx,num_threads,bounds(1:2)
! DEBUG
real(8) :: t1,t2,t3,t4,t5
! DEBUG
           
            !* numerical param - size of buffer for ultracell (MB)
            buffersizeMB = 10.0d0

            !* buffer size in 3x64b floats
            buffersizeDBLS = int(buffersizeMB*dble(1e6)/(3.0d0*64.0d0))
            
            if (updating_features) then
                !* only recomputing features benefits from storing all neigh info
                call activate_performance_option("keep_all_neigh_info")
            end if

            !* max cut off of all interactions
            mxrcut = maxrcut(0)
            
            !* whether twobody interactions are present
            calc_twobody = twobody_features_present()

            !* whether threebody interactions are present
            calc_threebody = threebody_features_present()
            
            !* set whether or not to calculate feature derivatives
            if (need_forces) then
                call switch_property("forces","on")
            else
                call switch_property("forces","off")
            end if
            if (need_stress) then
                call switch_property("forces","on")
                call switch_property("stress","on")
            else
                call switch_property("stress","off")
            end if

            if (speedup_applies("keep_all_neigh_info")) then
                if (allocated(set_neigh_info).neqv..true.) then
                    !* this is the first time running through
                    call init_set_neigh_info(set_type)
                end if
            else
                !* if no info is kept, always need to recompute
                call init_set_neigh_info(set_type)
            end if
            
            if (parallel) then
                !$omp parallel num_threads(omp_get_max_threads()),&
                !$omp& default(shared),&
                !$omp& private(thread_idx,ultra_cart,bounds,num_threads),&
                !$omp& private(ultra_idx,ultra_z)

                !* thread_idx = [0,num_threads-1]    
                thread_idx = omp_get_thread_num()
                
                num_threads = omp_get_max_threads()

                !* assume all confs take equal computation time
                call load_balance_alg_1(thread_idx,num_threads,data_sets(set_type)%nconf,bounds)

                do conf=bounds(1),bounds(2),1

                    if (.not.allocated(set_neigh_info(conf)%twobody)) then
                        !* only calculate if first time running
                        call get_ultracell(mxrcut,buffersizeDBLS,set_type,conf,&
                                &ultra_cart,ultra_idx,ultra_z)

                        !* always calc. two-body info for features
                        call calculate_twobody_info(set_type,conf,ultra_cart,ultra_z,ultra_idx)
                    
                        if (calc_threebody) then
                            !* calc. threebody info
                            call calculate_threebody_info(set_type,conf,&
                                    &ultra_cart,ultra_z,ultra_idx)
                        end if
                   end if 
                    
                    !* calculate features and their derivatives
                    call calculate_all_features(set_type,conf,updating_features)
                    !call experimental_feature_calc(set_type,conf,updating_features)
                    !call shift_x(set_type,conf)
                    
                    ! deprecated v
                    if (allocated(ultra_z)) then
                        deallocate(ultra_z)
                        deallocate(ultra_idx)
                        deallocate(ultra_cart)
                    end if

                    if (.not.speedup_applies("keep_all_neigh_info")) then
                        deallocate(set_neigh_info(conf)%twobody)
                        if (calc_threebody) then
                            deallocate(set_neigh_info(conf)%threebody)
                        end if
                    end if
                end do !* end loop over configurations

                !$omp end parallel
            else
                do conf=1,data_sets(set_type)%nconf
                    if (.not.allocated(set_neigh_info(conf)%twobody)) then
! DEBUG
call cpu_time(t1)
! DEBUG
                        call get_ultracell(mxrcut,5000,set_type,conf,&
                                &ultra_cart,ultra_idx,ultra_z)
! DEBUG
call cpu_time(t2)
! DEBUG

                        !* always calc. two-body info for features
                        call calculate_twobody_info(set_type,conf,ultra_cart,ultra_z,ultra_idx)
! DEBUG
call cpu_time(t3)
! DEBUG
                
                        if (calc_threebody) then
                            !* calc. threebody info
                            call calculate_threebody_info(set_type,conf,&
                                    &ultra_cart,ultra_z,ultra_idx)
                        end if
! DEBUG
call cpu_time(t4)
! DEBUG
                    end if
                    !* calculate features and their derivatives
                    call calculate_all_features(set_type,conf,updating_features)
                    !call experimental_feature_calc(set_type,conf,updating_features)
                    !call shift_x(set_type,conf)
! DEBUG
call cpu_time(t5)
!write(*,*) ''
!write(*,*) 'get_ultracell     twobody_info    threebody_info    calc all features'
!write(*,*) t2-t1,t3-t2,t4-t3,t5-t4
! DEBUG
                    ! deprecated v
                    if (allocated(ultra_z)) then
                        deallocate(ultra_z)
                        deallocate(ultra_idx)
                        deallocate(ultra_cart)
                    end if
                    !deallocate(feature_isotropic)
                    !if (calc_threebody) then
                    !    deallocate(feature_threebody_info)
                    !end if
                    ! deprecated ^

                    if (.not.speedup_applies("keep_all_neigh_info")) then
                        deallocate(set_neigh_info(conf)%twobody)
                        if (calc_threebody) then
                            deallocate(set_neigh_info(conf)%threebody)
                        end if
                    end if
                end do !* end loop over configurations
            end if !* end if parallel section
            
            if (updating_features) then
                !* store pre computed neighbour info
                atom_neigh_info_needs_updating = .false.
            end if

            if (.not.speedup_applies("keep_all_neigh_info")) then
                !* lets not be misleading
                deallocate(set_neigh_info)
            end if

            if (scale_features) then
                call scale_set_features(set_type)
            end if
        end subroutine calculate_features_singleset

        subroutine calculate_distance_distributions(set_type,sample_rate,mask,twobody_dist,&
        &threebody_dist,num_two,num_three)
            implicit none

            !* args
            integer,intent(in) :: set_type
            real(8),intent(in) :: sample_rate(1:2)
            logical,intent(in) :: mask(:)
            real(8),intent(out) :: twobody_dist(:)
            real(8),intent(out) :: threebody_dist(:,:)
            integer,intent(out) :: num_two,num_three


            !* scratch
            integer :: conf,atm,bond,dim_1,dim_2(1:2)
            integer :: cntr2,cntr3
            real(8) :: mxrcut
            real(8),allocatable :: ultra_cart(:,:)
            real(8),allocatable :: ultra_z(:)
            integer,allocatable :: ultra_idx(:)
            logical :: calc_threebody

            !* max cut off of all interactions
            mxrcut = maxrcut(0)
            
            !* whether threebody interactions are present
            calc_threebody = threebody_features_present()

            !* number of two and threebody terms output
            num_two = 0
            num_three = 0

            !* shape of input array
            dim_1 = size(twobody_dist)
            dim_2 = shape(threebody_dist)
           
            if (allocated(set_neigh_info)) then
                deallocate(set_neigh_info)
            end if
            allocate(set_neigh_info(data_sets(set_type)%nconf))
            
            cntr2 = 0
            cntr3 = 0
            do conf=1,data_sets(set_type)%nconf
                call get_ultracell(mxrcut,5000,set_type,conf,&
                        &ultra_cart,ultra_idx,ultra_z)

                !* always calc. two-body info for features
                call calculate_twobody_info(set_type,conf,ultra_cart,ultra_z,ultra_idx)
            
                do atm=1,data_sets(set_type)%configs(conf)%n
                    cntr2 = cntr2 + 1
                    if (cntr2.gt.size(mask)) then
                        call error("calculate_distance_distribution","mask array is too small")
                    end if                            
                    if (mask(cntr2)) then
                        !* allows custom query of specific atom environments
                        cycle
                    end if
                    
                    if (set_neigh_info(conf)%twobody(atm)%n.gt.0) then
                        do bond=1,set_neigh_info(conf)%twobody(atm)%n,1
                            if ((abs(sample_rate(1)-1.0d0).lt.1e-10).or.&
                            &(rand().lt.sample_rate(1))) then
                                !* book keeping
                                num_two = num_two + 1
                               
                                if (num_two.gt.dim_1) then
                                    call error("calculate_distance_distributions",&
                                            &"two-body buffer too small, increase or decrease sample rate.")
                                end if
                                
                                !* store atom-atom distance from this bond
                                twobody_dist(num_two) = set_neigh_info(conf)%twobody(atm)%dr(bond)
                            end if
                        end do
                    end if
                end do

                if (calc_threebody) then
                    !* calc. threebody info
                    call calculate_threebody_info(set_type,conf,ultra_cart,ultra_z,ultra_idx)
                
                    do atm=1,data_sets(set_type)%configs(conf)%n
                        cntr3 = cntr3 + 1
                        if (cntr3.gt.size(mask)) then
                            call error("calculate_distance_distribution","mask array is too small")
                        end if                            
                        if (mask(cntr3)) then
                            !* allows custom query of specific atom environments
                            cycle
                        end if
                        
                        if (set_neigh_info(conf)%threebody(atm)%n.gt.0) then
                            do bond=1,set_neigh_info(conf)%threebody(atm)%n,1
                                if ((abs(sample_rate(2)-1.0d0).lt.1e-10).or.&
                                &(rand().lt.sample_rate(2))) then
                                    !* book keeping
                                    num_three = num_three + 1
                                
                                    if (num_three.gt.dim_2(2)) then
                                        call error("calculate_distance_distributions",&
                                            &"three-body buffer too small, increase or decrease sample rate.")
                                    end if

                                    threebody_dist(1:2,num_three) = set_neigh_info(conf)%&
                                            &threebody(atm)%dr(1:2,bond)
                                    threebody_dist(3,num_three) = set_neigh_info(conf)%&
                                            &threebody(atm)%cos_ang(bond)
                                end if
                            end do !* end loop over bonds
                        end if 
                    end do !* loop over local atoms
                end if
                   
                                  
                if (allocated(ultra_z)) then   
                    deallocate(ultra_z)
                    deallocate(ultra_idx)
                    deallocate(ultra_cart)
                end if

                if (speedup_applies("keep_all_neigh_info").neqv..true.) then
                    deallocate(set_neigh_info(conf)%twobody)
                    if (calc_threebody) then
                        deallocate(set_neigh_info(conf)%threebody)
                    end if
                end if
            end do !* end loop over confs
        end subroutine calculate_distance_distributions

        subroutine calculate_all_features(set_type,conf,updating_features)
            implicit none

            integer,intent(in) :: set_type,conf
            logical,intent(in) :: updating_features
        
            !* scratch
            integer :: atm,ft
            
            if (updating_features) then
                !* when called while peforming opt. of basis func. params
                updating_net_weights_only = .false.
            end if

            do atm=1,data_sets(set_type)%configs(conf)%n
                do ft=1,feature_params%num_features
                    call evaluate_feature(ft,set_type,conf,atm)
                end do !* end loop features

                !* null dimension for bias
                data_sets(set_type)%configs(conf)%x(1,atm) = 1.0d0
            end do !* end loop atoms
        end subroutine calculate_all_features

        subroutine evaluate_feature(idx,set_type,conf,atm)
            implicit none

            integer,intent(in) :: idx,set_type,conf,atm

            integer :: ftype
! DEBUG
real(8) :: t1,t2
! DEBUG
            !* feature int id
            ftype = feature_params%info(idx)%ftype
            
            if (ftype.eq.featureID_StringToInt("atomic_number")) then
                call feature_atomicnumber_1(set_type,conf,atm,idx)
            else if (feature_IsTwoBody(ftype)) then
! DEBUG
call cpu_time(t1)   
!DEBUG             
                call feature_twobody(set_type,conf,atm,idx)
! DEBUG
call cpu_time(t2) 
!write(*,*) '2body feat comp:',t2-t1        
! DEBUG 
            else 
! DEBUG
call cpu_time(t1)   
!DEBUG             
                call feature_threebody(set_type,conf,atm,idx)
! DEBUG
call cpu_time(t2) 
!write(*,*) '3body feat comp:',t2-t1        
! DEBUG 
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
            integer :: arr_idx ,ii,cntr,arg,ftype
            integer :: contrib_atms(1:data_sets(set_type)%configs(conf)%n)
            integer :: idx_to_contrib
            integer :: ftype_acsf_behler_g1,ftype_acsf_behler_g2
            logical :: zero_neighbours 
            logical :: property_calc_forces,property_calc_stress
            logical :: speedup_twobody_rcut,speedup_single_element
            real(8) :: rcut,tmpz
            
            zero_neighbours = .true.

            !* weight bias takes 1st element of feature vector
            arr_idx = ft_idx + 1

            !* interaction cut off
            rcut = feature_params%info(ft_idx)%rcut

            !* type of interaction
            ftype = feature_params%info(ft_idx)%ftype
            
            if (atom_neigh_info_needs_updating) then
                !* when recomputing features for fixed cell,positions, can store this info
                
                if (set_neigh_info(conf)%twobody(atm)%n.gt.0) then
                    do ii=1,set_neigh_info(conf)%twobody(atm)%n,1
                        !* search for neighbour with cut off radius
                        if (set_neigh_info(conf)%twobody(atm)%dr(ii).le.rcut) then
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
            
                !* idx map associates supercell image atoms with their local cell copy
                if (allocated(data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%idx_map)) then
                    deallocate(data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%idx_map)
                end if
                allocate(data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%idx_map(&
                        &1:set_neigh_info(conf)%twobody(atm)%n,1:1))

                !* idx of central atom
                contrib_atms(1) = atm  

                cntr = 1

                do ii=1,set_neigh_info(conf)%twobody(atm)%n,1
                    if ( int_in_intarray(set_neigh_info(conf)%twobody(atm)%idx(ii),&
                    &contrib_atms(1:cntr),arg) ) then
                        !* Local atom already in list, note corresponding idx in contrib_atms
                        data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%idx_map(ii,1) = arg
                        cycle
                    else if (set_neigh_info(conf)%twobody(atm)%dr(ii).le.rcut) then
                        cntr = cntr + 1
                        !* note this local atom contributes to this feature for atom
                        contrib_atms(cntr) = set_neigh_info(conf)%twobody(atm)%idx(ii)
                        data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%idx_map(ii,1) = cntr
                    else
                        !* atom is beyond interaction cut off
                        data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%idx_map(ii,1) = -1
                    end if
                end do !* end loop over neighbour images
              
                !* in feature selection, feature computation is iterated over 
                if (allocated(data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%idx)) then
                    deallocate(data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%idx)
                end if
                if (allocated(data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec)) then
                    deallocate(data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec)
                end if
                if (allocated(data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%stress)) then
                    deallocate(data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%stress)
                end if

                allocate(data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%idx(cntr))
                allocate(data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(3,cntr))
                if (calculate_property("stress")) then
                    !* stress contributions of non-local neighbours
                    allocate(data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%stress(3,3,cntr))
                end if
                
                !* number of atoms in local cell contributing to feature (including central atom)
                data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%n = cntr
                
                !* local indices of atoms contributing to feature
                data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%idx(:) = contrib_atms(1:cntr)
            
            end if
            
            !* zero features
            data_sets(set_type)%configs(conf)%x(arr_idx,atm) = 0.0d0
            data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(:,:) = 0.0d0
            if (calculate_property("stress")) then
                data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%stress(:,:,:) = 0.0d0
            end if
           
            !* logical comparison quicker than string
            property_calc_forces = calculate_property("forces")
            property_calc_stress = calculate_property("stress")
            speedup_twobody_rcut = speedup_applies("twobody_rcut")
            speedup_single_element = speedup_applies("single_element")
           
            !* int comparison quicker than string
            ftype_acsf_behler_g1 = featureID_StringToInt("acsf_behler-g1")
            ftype_acsf_behler_g2 = featureID_StringToInt("acsf_behler-g2")
          
            !* atomic weighting method call string comparison, quicker to call just once
            tmpz = atomic_weighting(-1.0d0,-1.0d0,-1.0d0,ft_idx)

! debug
!return
! debug
            
            do ii=1,set_neigh_info(conf)%twobody(atm)%n
                if (set_neigh_info(conf)%twobody(atm)%dr(ii).le.rcut) then
                    !* contributions from supercell atoms must be summed with their local cell copy
                    idx_to_contrib = data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%&
                            &idx_map(ii,1)
                    
                    !* contributing interaction
                    if (ftype.eq.ftype_acsf_behler_g1) then
                        call feature_behler_g1(conf,atm,ii,ft_idx,&
                                &data_sets(set_type)%configs(conf)%x(arr_idx,atm))

                        if (calculate_property("forces")) then
                            call feature_behler_g1_deriv(set_type,conf,atm,ii,ft_idx,&
                                &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%&
                                &vec(1:3,idx_to_contrib),idx_to_contrib)
                        end if
                    else if (ftype.eq.ftype_acsf_behler_g2) then
                        call feature_behler_g2(conf,atm,ii,ft_idx,speedup_twobody_rcut,&
                                &speedup_single_element,tmpz,&
                                &data_sets(set_type)%configs(conf)%x(arr_idx,atm))

                        if (property_calc_forces) then
                            call feature_behler_g2_deriv(set_type,conf,atm,ii,ft_idx,&
                                &speedup_twobody_rcut,speedup_single_element,property_calc_stress,&
                                &tmpz,&
                                &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%&
                                &vec(1:3,idx_to_contrib),idx_to_contrib)
                        end if
                    else if (ftype.eq.featureID_StringToInt("acsf_normal-b2")) then
                        call feature_normal_iso(conf,atm,ii,ft_idx,&
                                &data_sets(set_type)%configs(conf)%x(arr_idx,atm))

                        if (calculate_property("forces")) then
                            call feature_normal_iso_deriv(set_type,conf,atm,ii,ft_idx,&
                                    &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%&
                                    &vec(1:3,idx_to_contrib),idx_to_contrib)
                        end if
                    else if (ftype.eq.featureID_StringToInt("acsf_fourier-b2")) then
                        call feature_fourier_b2(conf,atm,ii,ft_idx,&
                                &data_sets(set_type)%configs(conf)%x(arr_idx,atm))

                        if (calculate_property("forces")) then
                            call feature_fourier_b2_deriv(set_type,conf,atm,ii,ft_idx,&
                                    &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%&
                                    &vec(1:3,idx_to_contrib),idx_to_contrib)
                        end if
                    else if (ftype.eq.featureID_StringToInt("devel_iso")) then
                        call feature_iso_devel(conf,atm,ii,ft_idx,&
                                &data_sets(set_type)%configs(conf)%x(arr_idx,atm))
                        
                        if (calculate_property("forces")) then
                            call feature_iso_devel_deriv(set_type,conf,atm,ii,ft_idx,&
                                    &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%&
                                    &vec(1:3,idx_to_contrib),idx_to_contrib)
                        end if
                    end if
                end if
            end do
            
            !* derivative wrt. central atm
            if (calculate_property("forces")) then
                if (ftype.eq.featureID_StringToInt("acsf_behler-g1")) then
                    call feature_behler_g1_deriv(set_type,conf,atm,0,ft_idx,&
                            &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(1:3,1),1)
                else if (ftype.eq.featureID_StringToInt("acsf_behler-g2")) then
                    call feature_behler_g2_deriv(set_type,conf,atm,0,ft_idx,&
                            &speedup_twobody_rcut,speedup_single_element,property_calc_stress,&
                            &tmpz,&
                            &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(1:3,1),1)
                else if (ftype.eq.featureID_StringToInt("acsf_normal-b2")) then
                    call feature_normal_iso_deriv(set_type,conf,atm,0,ft_idx,&
                            &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(1:3,1),1)
                else if (ftype.eq.featureID_StringToInt("acsf_fourier-b2")) then
                    call feature_fourier_b2_deriv(set_type,conf,atm,0,ft_idx,&
                            &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(1:3,1),1)
                else if (ftype.eq.featureID_StringToInt("devel_iso")) then
                    call feature_iso_devel_deriv(set_type,conf,atm,0,ft_idx,&
                            &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(1:3,1),1)
                end if
            end if
        end subroutine feature_twobody

        subroutine feature_threebody(set_type,conf,atm,ft_idx)
            implicit none

            integer,intent(in) :: set_type,conf,atm,ft_idx

            real(8) :: rcut
            integer :: ii,jj,arr_idx,cntr,ftype
            integer :: contrib_atms(1:data_sets(set_type)%configs(conf)%n)
            integer :: arg
            logical :: not_null
            logical,allocatable :: bond_contributes(:)
! DEBUG
real(8) :: t1,t2,t3,t4
call cpu_time(t1)
! DEBUG
            rcut = feature_params%info(ft_idx)%rcut

            !* weight bias included by null feature coordinate
            arr_idx = ft_idx + 1

            not_null = .false.
            
            !* feature type
            ftype = feature_params%info(ft_idx)%ftype

            !* is three-body term within rcut?
            allocate(bond_contributes(set_neigh_info(conf)%threebody(atm)%n))
            
            if (feat_doesnt_taper_drjk(ft_idx)) then
                do ii=1,set_neigh_info(conf)%threebody(atm)%n,1
                    if (maxval(set_neigh_info(conf)%threebody(atm)%dr(1:2,ii)).le.rcut) then
                        bond_contributes(ii) = .true.
                    else
                        bond_contributes(ii) = .false.
                    end if
                end do
            else
                do ii=1,set_neigh_info(conf)%threebody(atm)%n,1
                    if (maxval(set_neigh_info(conf)%threebody(atm)%dr(1:3,ii)).le.rcut) then
                        bond_contributes(ii) = .true.
                    else
                        !* since drjk is tapered, interaction is 0 for drjk > rcut
                        bond_contributes(ii) = .false.
                    end if
                end do !* end loop over neighbours
            end if
            if ( (any(bond_contributes).neqv..true.).or.&
            &(set_neigh_info(conf)%threebody(atm)%n.eq.0) ) then
                !* zero neighbours within rcut
                data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%n=0
                data_sets(set_type)%configs(conf)%x(arr_idx,atm) = 0.0d0
                return
            end if
            
! DEBUG
call cpu_time(t2)
! this section is slow vv CAN FACTORISE THIS IF ALL 2body and 3body features 
! share same cut off radii
! DEBUG            
            if (atom_neigh_info_needs_updating) then
                !* re compute this when chaning rcut of data set
                
                if (allocated(data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%idx)) then
                    deallocate(data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%idx)
                end if
                if (allocated(data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec)) then
                    deallocate(data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec)
                end if
                if (allocated(data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%stress)) then
                    deallocate(data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%stress)
                end if
                if (allocated(data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%idx_map)) then
                    deallocate(data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%idx_map)
                end if

                !* idx of central atom
                contrib_atms(1) = atm  


                allocate(data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%idx_map(1:2,&
                        &1:set_neigh_info(conf)%threebody(atm)%n))
                
                !* NULL value for terms not within rcut
                data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%idx_map(:,:) = -1

                cntr = 1
                do ii=1,set_neigh_info(conf)%threebody(atm)%n,1
                    if (bond_contributes(ii).neqv..true.) then
                        cycle
                    end if

                    do jj=1,2
                        if ( int_in_intarray(set_neigh_info(conf)%threebody(atm)%idx(jj,ii),&
                        &contrib_atms(1:cntr),arg) ) then
                            !* Local atom already in list, note corresponding idx in contrib_atms
                            data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%&
                                    &idx_map(jj,ii) = arg
                            cycle
                        else 
                            cntr = cntr + 1
                            !* note this local atom contributes to this feature for atom
                            contrib_atms(cntr) = set_neigh_info(conf)%threebody(atm)%idx(jj,ii)
                            
                            data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%&
                                    &idx_map(jj,ii) = cntr
                        end if
                    end do !* end loop over 2 neighbouring atoms                    
                end do !* end loop over threebody terms
            
                allocate(data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%idx(cntr))
                allocate(data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(3,cntr))
                if (calculate_property("stress")) then
                    allocate(data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%stress(3,3,cntr))
                end if
                
                !* number of atoms in local cell contributing to feature (including central atom)
                data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%n = cntr
                
                !* local indices of atoms contributing to feature
                data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%idx(:) = contrib_atms(1:cntr)
            end if !* update idx,idx_map
! DEBUG
call cpu_time(t3)
! THIS section is slow vv
! DEBUG

            
            !* zero features
            data_sets(set_type)%configs(conf)%x(arr_idx,atm) = 0.0d0
            data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(:,:) = 0.0d0
            if (calculate_property("stress")) then
                data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%stress(:,:,:) = 0.0d0
            end if

            !do ii=1,feature_threebody_info(atm)%n,1
            do ii=1,set_neigh_info(conf)%threebody(atm)%n,1
                if(bond_contributes(ii).neqv..true.) then
                    cycle
                end if

                if (ftype.eq.featureID_StringToInt("acsf_behler-g4")) then
                    call feature_behler_g4(set_type,conf,atm,ft_idx,ii)

                    if (calculate_property("forces")) then
                        call feature_behler_g4_deriv(set_type,conf,atm,ft_idx,ii,&
                            &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%idx_map(:,ii)) 
                    end if
                else if (ftype.eq.featureID_StringToInt("acsf_behler-g5")) then
                    call feature_behler_g5(set_type,conf,atm,ft_idx,ii)

                    if (calculate_property("forces")) then
                        call feature_behler_g5_deriv(set_type,conf,atm,ft_idx,ii,&
                            &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%idx_map(:,ii))
                    end if
                else if (ftype.eq.featureID_StringToInt("acsf_normal-b3")) then
                    call feature_normal_threebody(set_type,conf,atm,ft_idx,ii)

                    if (calculate_property("forces")) then
                        call feature_normal_threebody_deriv(set_type,conf,atm,ft_idx,ii,&
                            &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%idx_map(:,ii))
                    end if
                end if
            end do !* end loop ii over three body terms
            deallocate(bond_contributes)

            if (updating_net_weights_only) then
                !* only useful for recomputing loss wrt basis function params
                deallocate(data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%idx_map)
            end if
! DEBUG
call cpu_time(t4)
!write(*,*) t2-t1,t3-t2,t4-t3
! DEBUG       
        end subroutine feature_threebody
        
        subroutine feature_behler_g1(conf,atm,neigh_idx,ft_idx,current_val)
            implicit none

            integer,intent(in) :: conf,atm,neigh_idx,ft_idx
            real(8),intent(inout) :: current_val

            !* scratch
            real(8) :: dr,tmp2,tmp3,za,zb,rcut,fs
            real(8) :: zatom,zneigh

            !* atom-neigh_idx distance 
            dr = set_neigh_info(conf)%twobody(atm)%dr(neigh_idx)

            !* symmetry function params
            za   = feature_params%info(ft_idx)%za
            zb   = feature_params%info(ft_idx)%zb
            fs   = feature_params%info(ft_idx)%fs
            rcut = feature_params%info(ft_idx)%rcut


            if (dr.gt.rcut) then
                current_val = current_val
                return
            end if

            !* tapering
            if (speedup_applies("twobody_rcut")) then
                tmp2 = set_neigh_info(conf)%twobody(atm)%dr_taper(neigh_idx)
            else
                tmp2 = taper_1(dr,rcut,fs)
            end if
       
            if (speedup_applies("single_element")) then
                tmp3 = atomic_weighting(-1.0d0,-1.0d0,-1.0d0,ft_idx)
            else
                zatom = set_neigh_info(conf)%twobody(atm)%z_atom
                zneigh = set_neigh_info(conf)%twobody(atm)%z(neigh_idx)

                tmp3 = atomic_weighting(zatom,zneigh,-1.0d0,ft_idx)
            end if
            
            current_val = current_val + tmp2*tmp3
        end subroutine feature_behler_g1
        
        subroutine feature_behler_g1_deriv(set_type,conf,atm,neigh_idx,ft_idx,deriv_vec,deriv_idx)
            implicit none

            integer,intent(in) :: set_type,conf,atm,neigh_idx,ft_idx,deriv_idx
            real(8),intent(inout) :: deriv_vec(1:3)

            !* scratch
            real(8) :: dr_scl,dr_vec(1:3),tap_deriv,tap,tmp1,tmp2
            real(8) :: fs,rcut,tmpz
            real(8) :: za,zb,deriv_nl_r(1:3)
            real(8) :: zatom,zneigh
            integer :: ii,lim1,lim2

            !* symmetry function params
            za   = feature_params%info(ft_idx)%za
            zb   = feature_params%info(ft_idx)%zb
            fs   = feature_params%info(ft_idx)%fs
            rcut = feature_params%info(ft_idx)%rcut
           
            if (neigh_idx.eq.0) then
                lim1 = 1
                lim2 = set_neigh_info(conf)%twobody(atm)%n
                tmp2 = -1.0d0       !* sign for drij/d r_central
                if (calculate_property("stress")) then
                    deriv_nl_r = set_neigh_info(conf)%twobody(atm)%r_nl_atom
                end if
            else
                lim1 = neigh_idx
                lim2 = neigh_idx    
                tmp2 = 1.0d0        !* sign for drij/d r_neighbour
                if (calculate_property("stress")) then 
                    deriv_nl_r = set_neigh_info(conf)%twobody(atm)%r_nl_neigh(:,lim1)
                end if
            end if
            
            if (speedup_applies("single_element")) then
                tmpz = atomic_weighting(-1.0d0,-1.0d0,-1.0d0,ft_idx)
            end if


            !* derivative wrt. central atom itself
            do ii=lim1,lim2,1
                ! NO SELF INTERACTION
                if (atm.eq.set_neigh_info(conf)%twobody(atm)%idx(ii)) then
                    ! dr_vec =  d (r_i + const - r_i ) / d r_i = 0
                    cycle
                end if
                
                !* atom-atom distance
                dr_scl = set_neigh_info(conf)%twobody(atm)%dr(ii)
                
                if (dr_scl.gt.rcut) then
                    cycle
                end if

                !* (r_neighbour - r_centralatom)/dr_scl
                dr_vec(:) = set_neigh_info(conf)%twobody(atm)%drdri(:,ii)
                
                !* tapering
                if (speedup_applies("twobody_rcut")) then
                    tap = set_neigh_info(conf)%twobody(atm)%dr_taper(ii)
                    tap_deriv = set_neigh_info(conf)%twobody(atm)%dr_taper_deriv(ii)
                else
                    tap = taper_1(dr_scl,rcut,fs)
                    tap_deriv = taper_deriv_1(dr_scl,rcut,fs)
                end if

                if (.not.speedup_applies("single_element")) then
                    zatom = set_neigh_info(conf)%twobody(atm)%z_atom
                    zneigh = set_neigh_info(conf)%twobody(atm)%z(ii)

                    tmpz = atomic_weighting(zatom,zneigh,-1.0d0,ft_idx)
                end if

                tmp1 = tap_deriv
               
                !* cumulative atom derivative for given feature 
                deriv_vec(:) = deriv_vec(:) + dr_vec(:)*tmp1*tmp2*tmpz

                if (calculate_property("stress")) then
                    call append_stress_contribution(dr_vec*tmp1*tmp2*tmpz,deriv_nl_r,&
                            &set_type,conf,atm,ft_idx,deriv_idx)
                end if
            end do
        end subroutine feature_behler_g1_deriv
        
        subroutine feature_behler_g2(conf,atm,neigh_idx,ft_idx,speedup_twobody_rcut,&
        &speedup_single_element,tmpz,current_val)
            implicit none

            integer,intent(in) :: conf,atm,neigh_idx,ft_idx
            logical,intent(in) :: speedup_twobody_rcut,speedup_single_element
            real(8) :: tmpz
            real(8),intent(inout) :: current_val

            !* scratch
            real(8) :: dr,tmp1,tmp2,tmp3,za,zb,rcut,eta,rs,fs
            real(8) :: zatom,zneigh
           
            !* atom-neigh_idx distance 
            dr = set_neigh_info(conf)%twobody(atm)%dr(neigh_idx)

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
            if (speedup_twobody_rcut) then
                tmp2 = set_neigh_info(conf)%twobody(atm)%dr_taper(neigh_idx)
            else
                tmp2 = taper_1(dr,rcut,fs)
            end if

            !* atomic number        
            if (speedup_single_element) then
                ! debug
                tmp3 = tmpz
                ! debug
                !tmp3 = atomic_weighting(-1.0d0,-1.0d0,-1.0d0,ft_idx)
            else
                zatom = set_neigh_info(conf)%twobody(atm)%z_atom
                zneigh = set_neigh_info(conf)%twobody(atm)%z(neigh_idx)
                tmp3 = atomic_weighting(zatom,zneigh,-1.0d0,ft_idx)
            end if

            current_val = current_val + tmp1*tmp2*tmp3
        end subroutine feature_behler_g2
      
        subroutine feature_behler_g2_deriv(set_type,conf,atm,neigh_idx,ft_idx,&
        &speedup_twobody_rcut,speedup_single_element,calculate_property_stress,tmpz,&
        &deriv_vec,deriv_idx)
            implicit none

            integer,intent(in) :: set_type,conf,atm,neigh_idx,ft_idx,deriv_idx
            logical,intent(in) :: speedup_twobody_rcut,speedup_single_element
            logical,intent(in) :: calculate_property_stress
            real(8),intent(in) :: tmpz
            real(8),intent(inout) :: deriv_vec(1:3)

            !* scratch
            real(8) :: dr_scl,dr_vec(1:3),tap_deriv,tap,tmp1,tmp2
            real(8) :: rs,fs,rcut,tmpz_scratch
            real(8) :: za,zb,eta,r_nl(1:3)
            real(8) :: zatom,zneigh
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
                lim2 = set_neigh_info(conf)%twobody(atm)%n
                tmp2 = -1.0d0       !* sign for drij/d r_central
                if (calculate_property_stress) then
                    r_nl = set_neigh_info(conf)%twobody(atm)%r_nl_atom
                end if
            else
                lim1 = neigh_idx
                lim2 = neigh_idx    
                tmp2 = 1.0d0        !* sign for drij/d r_neighbour
                if (calculate_property_stress) then
                    r_nl = set_neigh_info(conf)%twobody(atm)%r_nl_neigh(:,lim1)
                end if
            end if
            
            !* atomic number        
            if (speedup_single_element) then
                tmpz_scratch = tmpz
            end if


            !* derivative wrt. central atom itself
            do ii=lim1,lim2,1
! no self interaction          
!                if (atm.eq.set_neigh_info(conf)%twobody(atm)%idx(ii)) then
!                    ! dr_vec =  d (r_i + const - r_i ) / d r_i = 0
!                    cycle
!                end if
! debug       
                
                !* atom-atom distance
                dr_scl = set_neigh_info(conf)%twobody(atm)%dr(ii)

                !* (r_neighbour - r_centralatom)/dr_scl
                dr_vec(:) = set_neigh_info(conf)%twobody(atm)%drdri(:,ii)
                
                !* tapering
                if (speedup_twobody_rcut) then
                    tap = set_neigh_info(conf)%twobody(atm)%dr_taper(ii)
                    tap_deriv = set_neigh_info(conf)%twobody(atm)%dr_taper_deriv(ii)
                else
                    tap = taper_1(dr_scl,rcut,fs)
                    tap_deriv = taper_deriv_1(dr_scl,rcut,fs)
                end if

                if (.not.speedup_single_element) then
                    zatom = set_neigh_info(conf)%twobody(atm)%z_atom
                    zneigh = set_neigh_info(conf)%twobody(atm)%z(ii)
                    tmpz_scratch = atomic_weighting(zatom,zneigh,-1.0d0,ft_idx)
                end if

                tmp1 =  exp(-eta*(dr_scl-rs)**2)  *  (tap_deriv - &
                        &2.0d0*eta*(dr_scl-rs)*tap) 
               

                
                deriv_vec(:) = deriv_vec(:) + dr_vec(:)*tmp1*tmp2*tmpz_scratch
                
                if (calculate_property_stress) then
                    call append_stress_contribution(dr_vec*tmp1*tmp2*tmpz_scratch,r_nl,&
                            &set_type,conf,atm,ft_idx,deriv_idx)
                end if
            end do
        end subroutine feature_behler_g2_deriv
       
        subroutine feature_behler_g4(set_type,conf,atm,ft_idx,bond_idx)
            implicit none

            !* args
            integer,intent(in) :: atm,ft_idx,bond_idx,set_type,conf

            !* scratch
            real(8) :: xi,eta,lambda,fs,rcut,za,zb
            real(8) :: tmp_atmz,tmp_taper,zatom,zneigh1,zneigh2
            real(8) :: drij,drik,drjk,cos_angle

            !* feature parameters
            rcut   = feature_params%info(ft_idx)%rcut
            eta    = feature_params%info(ft_idx)%eta
            xi     = feature_params%info(ft_idx)%xi
            lambda = feature_params%info(ft_idx)%lambda
            fs     = feature_params%info(ft_idx)%fs
            za     = feature_params%info(ft_idx)%za
            zb     = feature_params%info(ft_idx)%zb

            !* atom-atom distances
            drij = set_neigh_info(conf)%threebody(atm)%dr(1,bond_idx)
            drik = set_neigh_info(conf)%threebody(atm)%dr(2,bond_idx)
            drjk = set_neigh_info(conf)%threebody(atm)%dr(3,bond_idx)

            if ( (drij.gt.rcut).or.(drik.gt.rcut).or.(drjk.gt.rcut) ) then
                return
            end if

            cos_angle = set_neigh_info(conf)%threebody(atm)%cos_ang(bond_idx)

            if (speedup_applies("threebody_rcut")) then
                tmp_taper = product(set_neigh_info(conf)%threebody(atm)%dr_taper(1:3,bond_idx))
            else
                tmp_taper = taper_1(drij,rcut,fs)*taper_1(drik,rcut,fs)*taper_1(drjk,rcut,fs)
            end if

            if (speedup_applies("single_element")) then
                tmp_atmz = atomic_weighting(-1.0d0,-1.0d0,-1.0d0,ft_idx)
            else
                zatom = set_neigh_info(conf)%threebody(atm)%z_atom
                zneigh1 = set_neigh_info(conf)%threebody(atm)%z(1,bond_idx)
                zneigh2 = set_neigh_info(conf)%threebody(atm)%z(2,bond_idx)
               
                tmp_atmz = atomic_weighting(zatom,zneigh1,zneigh2,ft_idx) 
            end if

            data_sets(set_type)%configs(conf)%x(ft_idx+1,atm) = &
                    &data_sets(set_type)%configs(conf)%x(ft_idx+1,atm)&
                    &+ 2**(1-xi)*(1.0d0 + lambda*cos_angle)**xi * &
                    &exp(-eta*(drij**2+drik**2+drjk**2))*tmp_taper*tmp_atmz
        end subroutine feature_behler_g4
       
        subroutine feature_behler_g4_deriv(set_type,conf,atm,ft_idx,bond_idx,idx_to_contrib) 
            implicit none

            !* args
            integer,intent(in) :: set_type,conf,atm,ft_idx,bond_idx
            integer,intent(in) :: idx_to_contrib(1:2)
            
            !* scratch
            real(8) :: xi,eta,lambda,fs,rcut,za,zb
            real(8) :: drij,drik,drjk,cos_angle,tmp_z,dxdr(1:3)
            integer :: zz,deriv_idx
            real(8) :: tmp_feature1,tmp_feature2,tap_ij,tap_jk,tap_ik
            real(8) :: tap_ij_deriv,tap_ik_deriv,tap_jk_deriv
            real(8) :: dcosdrz(1:3),drijdrz(1:3),drikdrz(1:3),drjkdrz(1:3)
            real(8) :: r_nl(1:3),zatom,zneigh1,zneigh2

            !* feature parameters
            rcut   = feature_params%info(ft_idx)%rcut
            eta    = feature_params%info(ft_idx)%eta
            xi     = feature_params%info(ft_idx)%xi
            lambda = feature_params%info(ft_idx)%lambda
            fs     = feature_params%info(ft_idx)%fs
            za     = feature_params%info(ft_idx)%za
            zb     = feature_params%info(ft_idx)%zb

            !* atom-atom distances
            drij = set_neigh_info(conf)%threebody(atm)%dr(1,bond_idx)
            drik = set_neigh_info(conf)%threebody(atm)%dr(2,bond_idx)
            drjk = set_neigh_info(conf)%threebody(atm)%dr(3,bond_idx)
            
            if ( (drij.gt.rcut).or.(drik.gt.rcut).or.(drjk.gt.rcut) ) then
                return
            end if

            cos_angle = set_neigh_info(conf)%threebody(atm)%cos_ang(bond_idx)

            !* tapering
            if (speedup_applies("threebody_rcut")) then
                !* same rcut,rs for all threebody features
                tap_ij = set_neigh_info(conf)%threebody(atm)%dr_taper(1,bond_idx)
                tap_ik = set_neigh_info(conf)%threebody(atm)%dr_taper(2,bond_idx)
                tap_jk = set_neigh_info(conf)%threebody(atm)%dr_taper(3,bond_idx)
                tap_ij_deriv = set_neigh_info(conf)%threebody(atm)%dr_taper_deriv(1,bond_idx)
                tap_ik_deriv = set_neigh_info(conf)%threebody(atm)%dr_taper_deriv(2,bond_idx)
                tap_jk_deriv = set_neigh_info(conf)%threebody(atm)%dr_taper_deriv(3,bond_idx)
            else
                tap_ij = taper_1(drij,rcut,fs)
                tap_ik = taper_1(drik,rcut,fs)
                tap_jk = taper_1(drjk,rcut,fs)
                tap_ij_deriv = taper_deriv_1(drij,rcut,fs)
                tap_ik_deriv = taper_deriv_1(drik,rcut,fs)
                tap_jk_deriv = taper_deriv_1(drjk,rcut,fs)
            end if

            if (speedup_applies("single_element")) then
                tmp_z = atomic_weighting(-1.0d0,-1.0d0,-1.0d0,ft_idx)
            else
                zatom = set_neigh_info(conf)%threebody(atm)%z_atom
                zneigh1 = set_neigh_info(conf)%threebody(atm)%z(1,bond_idx)
                zneigh2 = set_neigh_info(conf)%threebody(atm)%z(2,bond_idx)
               
                tmp_z = atomic_weighting(zatom,zneigh1,zneigh2,ft_idx) 
            end if

            tmp_feature1 = 2.0d0**(1.0d0-xi)*exp(-eta*(drij**2+drik**2+drjk**2)) * tmp_z
            tmp_feature2 = tmp_feature1 * (1.0d0+lambda*cos_angle)**xi
            

            ! 1=jj , 2=kk, 3=ii
            do zz=1,3,1
                ! map atom id to portion of mem for derivative
                if (zz.lt.3) then
                    deriv_idx = idx_to_contrib(zz) 
                else
                    deriv_idx = 1
                end if
                
                !* derivatives wrt r_zz
                dcosdrz =  set_neigh_info(conf)%threebody(atm)%dcos_dr(:,zz,bond_idx)
                
                if (zz.eq.1) then
                    ! zz=jj
                    drijdrz =  set_neigh_info(conf)%threebody(atm)%drdri(:,1,bond_idx)
                    drikdrz =  set_neigh_info(conf)%threebody(atm)%drdri(:,4,bond_idx)
                    drjkdrz = -set_neigh_info(conf)%threebody(atm)%drdri(:,5,bond_idx)
                else if (zz.eq.2) then
                    ! zz=kk
                    drijdrz =  set_neigh_info(conf)%threebody(atm)%drdri(:,2,bond_idx)
                    drikdrz =  set_neigh_info(conf)%threebody(atm)%drdri(:,3,bond_idx)
                    drjkdrz =  set_neigh_info(conf)%threebody(atm)%drdri(:,5,bond_idx)
                else if (zz.eq.3) then
                    ! zz=ii
                    drijdrz = -set_neigh_info(conf)%threebody(atm)%drdri(:,1,bond_idx)
                    drikdrz = -set_neigh_info(conf)%threebody(atm)%drdri(:,3,bond_idx)
                    drjkdrz =  set_neigh_info(conf)%threebody(atm)%drdri(:,6,bond_idx)
                end if
                if (calculate_property("stress")) then
                    r_nl = set_neigh_info(conf)%threebody(atm)%r_nl(:,zz,bond_idx)    
                end if


                dxdr = tap_ij*tap_ik*tap_jk*lambda*xi*((1.0d0+lambda*cos_angle)**(xi-1.0d0))*&
                    &dcosdrz*tmp_feature1 +&
                    &(tap_ik*tap_jk*(tap_ij_deriv - 2.0d0*eta*tap_ij*drij)*drijdrz +&
                    &tap_ij*tap_jk*(tap_ik_deriv - 2.0d0*eta*tap_ik*drik)*drikdrz +&
                    &tap_ij*tap_ik*(tap_jk_deriv - 2.0d0*eta*tap_jk*drjk)*drjkdrz  )*tmp_feature2

                data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(:,deriv_idx) = &
                    &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(:,deriv_idx) + dxdr
                
                if (calculate_property("stress")) then
                    call append_stress_contribution(dxdr,r_nl,set_type,conf,atm,ft_idx,deriv_idx)
                end if
            end do
            
        end subroutine feature_behler_g4_deriv
        
        subroutine feature_behler_g5(set_type,conf,atm,ft_idx,bond_idx)
            implicit none

            !* args
            integer,intent(in) :: atm,ft_idx,bond_idx,set_type,conf

            !* scratch
            real(8) :: xi,eta,lambda,fs,rcut,za,zb
            real(8) :: tmp_atmz,tmp_taper
            real(8) :: drij,drik,cos_angle
            real(8) :: zatom,zneigh1,zneigh2

            !* feature parameters
            rcut   = feature_params%info(ft_idx)%rcut
            eta    = feature_params%info(ft_idx)%eta
            xi     = feature_params%info(ft_idx)%xi
            lambda = feature_params%info(ft_idx)%lambda
            fs     = feature_params%info(ft_idx)%fs
            za     = feature_params%info(ft_idx)%za
            zb     = feature_params%info(ft_idx)%zb

            !* atom-atom distances
            drij = set_neigh_info(conf)%threebody(atm)%dr(1,bond_idx)
            drik = set_neigh_info(conf)%threebody(atm)%dr(2,bond_idx)
            
            if ( (drij.gt.rcut).or.(drik.gt.rcut) ) then
                return
            end if

            cos_angle = set_neigh_info(conf)%threebody(atm)%cos_ang(bond_idx)

            !* atomic number
            if (speedup_applies("single_element")) then
                tmp_atmz = atomic_weighting(-1.0d0,-1.0d0,-1.0d0,ft_idx)
            else
                zatom = set_neigh_info(conf)%threebody(atm)%z_atom
                zneigh1 = set_neigh_info(conf)%threebody(atm)%z(1,bond_idx)
                zneigh2 = set_neigh_info(conf)%threebody(atm)%z(2,bond_idx)
                
                tmp_atmz = atomic_weighting(zatom,zneigh1,zneigh2,ft_idx)
            end if

            !* taper term
            if (speedup_applies("threebody_rcut")) then
                tmp_taper = product(set_neigh_info(conf)%threebody(atm)%dr_taper(1:2,bond_idx))
            else
                tmp_taper = taper_1(drij,rcut,fs)*taper_1(drik,rcut,fs)
            end if

            data_sets(set_type)%configs(conf)%x(ft_idx+1,atm) = &
                    &data_sets(set_type)%configs(conf)%x(ft_idx+1,atm)&
                    &+ 2**(1-xi)*(1.0d0 + lambda*cos_angle)**xi * &
                    &exp(-eta*(drij**2+drik**2))*tmp_taper*tmp_atmz
        end subroutine feature_behler_g5
       
        subroutine feature_behler_g5_deriv(set_type,conf,atm,ft_idx,bond_idx,idx_to_contrib)                    
            implicit none

            !* args
            integer,intent(in) :: set_type,conf,atm,ft_idx,bond_idx
            integer,intent(in) :: idx_to_contrib(1:2)
            
            !* scratch
            real(8) :: xi,eta,lambda,fs,rcut,za,zb
            real(8) :: drij,drik,cos_angle,tmp_z
            integer :: zz,deriv_idx
            real(8) :: tmp_feature1,tmp_feature2,tap_ij,tap_ik
            real(8) :: tap_ij_deriv,tap_ik_deriv,r_nl(1:3),dxdr(1:3)
            real(8) :: dcosdrz(1:3),drijdrz(1:3),drikdrz(1:3)
            real(8) :: zatom,zneigh1,zneigh2

            !* feature parameters
            rcut   = feature_params%info(ft_idx)%rcut
            eta    = feature_params%info(ft_idx)%eta
            xi     = feature_params%info(ft_idx)%xi
            lambda = feature_params%info(ft_idx)%lambda
            fs     = feature_params%info(ft_idx)%fs
            za     = feature_params%info(ft_idx)%za
            zb     = feature_params%info(ft_idx)%zb

            !* atom-atom distances
            drij = set_neigh_info(conf)%threebody(atm)%dr(1,bond_idx)
            drik = set_neigh_info(conf)%threebody(atm)%dr(2,bond_idx)
            
            if ( (drij.gt.rcut).or.(drik.gt.rcut) ) then
                return
            end if

            cos_angle = set_neigh_info(conf)%threebody(atm)%cos_ang(bond_idx)

            !* tapering
            if (speedup_applies("threebody_rcut")) then
                tap_ij = set_neigh_info(conf)%threebody(atm)%dr_taper(1,bond_idx)
                tap_ik = set_neigh_info(conf)%threebody(atm)%dr_taper(2,bond_idx)
                tap_ij_deriv = set_neigh_info(conf)%threebody(atm)%dr_taper_deriv(1,bond_idx)
                tap_ik_deriv = set_neigh_info(conf)%threebody(atm)%dr_taper_deriv(2,bond_idx)
            else
                tap_ij = taper_1(drij,rcut,fs)
                tap_ik = taper_1(drik,rcut,fs)
                tap_ij_deriv = taper_deriv_1(drij,rcut,fs)
                tap_ik_deriv = taper_deriv_1(drik,rcut,fs)
            end if

            !* atomic number
            if (speedup_applies("single_element")) then
                tmp_z = atomic_weighting(-1.0d0,-1.0d0,-1.0d0,ft_idx)
            else
                zatom = set_neigh_info(conf)%threebody(atm)%z_atom
                zneigh1 = set_neigh_info(conf)%threebody(atm)%z(1,bond_idx)
                zneigh2 = set_neigh_info(conf)%threebody(atm)%z(2,bond_idx)
                
                tmp_z = atomic_weighting(zatom,zneigh1,zneigh2,ft_idx)
            end if

            tmp_feature1 = 2.0d0**(1.0d0-xi)*exp(-eta*(drij**2+drik**2))*tmp_z 
            tmp_feature2 = tmp_feature1 * (1.0d0+lambda*cos_angle)**xi
            
            ! 1=jj , 2=kk, 3=ii
            do zz=1,3,1
                ! map atom id to portion of mem for derivative
                if (zz.lt.3) then
                    deriv_idx = idx_to_contrib(zz) 
                else
                    deriv_idx = 1
                end if
                
                !* derivatives wrt r_zz
                dcosdrz =  set_neigh_info(conf)%threebody(atm)%dcos_dr(:,zz,bond_idx)
                
                if (zz.eq.1) then
                    ! zz=jj
                    drijdrz =  set_neigh_info(conf)%threebody(atm)%drdri(:,1,bond_idx)
                    drikdrz =  set_neigh_info(conf)%threebody(atm)%drdri(:,4,bond_idx)
                else if (zz.eq.2) then
                    ! zz=kk
                    drijdrz =  set_neigh_info(conf)%threebody(atm)%drdri(:,2,bond_idx)
                    drikdrz =  set_neigh_info(conf)%threebody(atm)%drdri(:,3,bond_idx)
                else if (zz.eq.3) then
                    ! zz=ii
                    drijdrz = -set_neigh_info(conf)%threebody(atm)%drdri(:,1,bond_idx)
                    drikdrz = -set_neigh_info(conf)%threebody(atm)%drdri(:,3,bond_idx)
                end if
                if (calculate_property("stress")) then
                    r_nl = set_neigh_info(conf)%threebody(atm)%r_nl(:,zz,bond_idx)    
                end if

                dxdr = tmp_feature1*tap_ij*tap_ik*lambda*xi*((1.0d0+lambda*cos_angle)**(xi-1.0d0))*&
                    &dcosdrz +&
                    &(tap_ik*(tap_ij_deriv - 2.0d0*eta*tap_ij*drij)*drijdrz +&
                    &tap_ij*(tap_ik_deriv - 2.0d0*eta*tap_ik*drik)*drikdrz )*tmp_feature2

                data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(:,deriv_idx) = &
                &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(:,deriv_idx) + dxdr 
                
                if (calculate_property("stress")) then
                    call append_stress_contribution(dxdr,r_nl,set_type,conf,atm,ft_idx,deriv_idx)
                end if
            end do
            
        end subroutine feature_behler_g5_deriv
        
        subroutine feature_normal_iso(conf,atm,neigh_idx,ft_idx,current_val)
            implicit none

            integer,intent(in) :: conf,atm,neigh_idx,ft_idx
            real(8),intent(inout) :: current_val

            !* scratch
            real(8) :: dr,tmp1,tmp2,tmp3,za,zb,rcut,fs,prec
            real(8) :: mean,sqrt_det
            real(8) :: zatom,zneigh

            !* atom-neigh_idx distance 
            dr  = set_neigh_info(conf)%twobody(atm)%dr(neigh_idx)
           
            !* symmetry function params
            za       = feature_params%info(ft_idx)%za
            zb       = feature_params%info(ft_idx)%zb
            fs       = feature_params%info(ft_idx)%fs
            prec     = feature_params%info(ft_idx)%prec(1,1)
            mean     = feature_params%info(ft_idx)%mean(1)
            rcut     = feature_params%info(ft_idx)%rcut
            sqrt_det = feature_params%info(ft_idx)%sqrt_det

            if (dr.gt.rcut) then
                return
            end if

            !* exponential
            tmp1 = exp(-0.5d0*prec*(dr-mean)**2)

            !* tapering
            if (speedup_applies("twobody_rcut")) then
                tmp2 = set_neigh_info(conf)%twobody(atm)%dr_taper(neigh_idx)
            else
                tmp2 = taper_1(dr,rcut,fs)
            end if
       
            if (speedup_applies("single_element")) then
                tmp3 = atomic_weighting(-1.0d0,-1.0d0,-1.0d0,ft_idx)
            else
                zatom = set_neigh_info(conf)%twobody(atm)%z_atom
                zneigh = set_neigh_info(conf)%twobody(atm)%z(neigh_idx)
                
                tmp3 = atomic_weighting(zatom,zneigh,-1.0d0,ft_idx) 
            end if

            current_val = current_val + tmp1*tmp2*tmp3
        end subroutine feature_normal_iso
        
        subroutine feature_normal_iso_deriv(set_type,conf,atm,neigh_idx,ft_idx,deriv_vec,deriv_idx)
            implicit none

            integer,intent(in) :: set_type,conf,atm,neigh_idx,ft_idx,deriv_idx
            real(8),intent(inout) :: deriv_vec(1:3)

            !* scratch
            real(8) :: dr_scl,dr_vec(1:3),tap_deriv,tap,tmp1,tmp2
            real(8) :: fs,rcut,tmpz,prec,mean,sqrt_det
            real(8) :: za,zb,invsqrt2pi,prec_const,r_nl(1:3)
            real(8) :: zatom,zneigh
            integer :: ii,lim1,lim2
            
            invsqrt2pi = 0.3989422804014327d0
            
            !* symmetry function params
            za       = feature_params%info(ft_idx)%za
            zb       = feature_params%info(ft_idx)%zb
            fs       = feature_params%info(ft_idx)%fs
            prec     = feature_params%info(ft_idx)%prec(1,1)
            mean     = feature_params%info(ft_idx)%mean(1)
            rcut     = feature_params%info(ft_idx)%rcut
            sqrt_det = feature_params%info(ft_idx)%sqrt_det

            prec_const = invsqrt2pi*sqrt_det
            
            if (neigh_idx.eq.0) then
                lim1 = 1
                lim2 = set_neigh_info(conf)%twobody(atm)%n
                tmp2 = -1.0d0       !* sign for drij/d r_central
                if (calculate_property("stress")) then
                    r_nl = set_neigh_info(conf)%twobody(atm)%r_nl_atom
                end if
            else
                lim1 = neigh_idx
                lim2 = neigh_idx    
                tmp2 = 1.0d0        !* sign for drij/d r_neighbour
                if (calculate_property("stress")) then
                    r_nl = set_neigh_info(conf)%twobody(atm)%r_nl_neigh(:,lim1)
                end if
            end if
            
            if (speedup_applies("single_element")) then
                tmpz = atomic_weighting(-1.0d0,-1.0d0,-1.0d0,ft_idx)
            end if                


            !* derivative wrt. central atom itself
            do ii=lim1,lim2,1
                ! NO SELF INTERACTION
                if (atm.eq.set_neigh_info(conf)%twobody(atm)%idx(ii)) then
                    ! dr_vec =  d (r_i + const - r_i ) / d r_i = 0
                    cycle
                end if
                
                !* atom-atom distance
                dr_scl = set_neigh_info(conf)%twobody(atm)%dr(ii)

                if (dr_scl.gt.rcut) then
                    cycle
                end if

                !* (r_neighbour - r_centralatom)/dr_scl
                dr_vec(:) = set_neigh_info(conf)%twobody(atm)%drdri(:,ii)
                
                !* tapering
                if (speedup_applies("twobody_rcut")) then
                    tap = set_neigh_info(conf)%twobody(atm)%dr_taper(ii)
                    tap_deriv = set_neigh_info(conf)%twobody(atm)%dr_taper_deriv(ii)
                else
                    tap = taper_1(dr_scl,rcut,fs)
                    tap_deriv = taper_deriv_1(dr_scl,rcut,fs)
                end if

                if (.not.speedup_applies("single_element")) then
                    zatom = set_neigh_info(conf)%twobody(atm)%z_atom
                    zneigh = set_neigh_info(conf)%twobody(atm)%z(ii)
                                       
                    tmpz = atomic_weighting(zatom,zneigh,-1.0d0,ft_idx) 
                end if

                tmp1 =  exp(-0.5d0*prec*(dr_scl-mean)**2)  *  (tap_deriv - &
                        &prec*(dr_scl-mean)*tap) 
                
                deriv_vec(:) = deriv_vec(:) + dr_vec(:)*tmp1*tmp2*tmpz
                
                if (calculate_property("stress")) then
                    call append_stress_contribution(dr_vec*tmp1*tmp2*tmpz,r_nl,&
                            &set_type,conf,atm,ft_idx,deriv_idx)
                end if
            end do
        end subroutine feature_normal_iso_deriv
        
        real(8) function func_normal(x,mean,prec)
            implicit none
            
            !* args
            real(8),intent(in) :: x(:),mean(:),prec(:,:)

            !* scratch
            integer :: n
            real(8) :: lwork(1:3),pi_const

            !* sqrt(1/(2 pi))
            pi_const = 0.3989422804014327

            n = size(x)

            !* lwork = prec * (x - mean) (prec is symmetric)
            call dsymv('u',n,1.0d0,prec,n,x-mean,1,0.0d0,lwork,1)

            !* sum_i lwork_i*(x-mean)_i
            func_normal = exp(-0.5d0*ddot(n,x-mean,1,lwork,1)) 
        end function func_normal

        subroutine feature_iso_devel(conf,atm,neigh_idx,ft_idx,current_val)
            use propagate, only : logistic

            implicit none

            integer,intent(in) :: conf,atm,neigh_idx,ft_idx
            real(8),intent(inout) :: current_val

            real(8) :: dr,rcuts(1:2),mean,const,r_taper
            real(8) :: tmp_taper,xtilde,fs,std

            mean = feature_params%info(ft_idx)%devel(1)
            const = feature_params%info(ft_idx)%devel(2)
            std = feature_params%info(ft_idx)%devel(3)
            fs = feature_params%info(ft_idx)%fs

            !* x-> const*(x-mean) up to x-mean = 2*std

            rcuts(1) = feature_params%info(ft_idx)%rcut
            rcuts(2) = mean + 2.0d0*std 

            r_taper = minval(rcuts)
            dr  = set_neigh_info(conf)%twobody(atm)%dr(neigh_idx)
            tmp_taper = taper_1(dr,r_taper,fs)

            xtilde = const*(dr-mean)
            
            current_val = current_val + logistic(xtilde)*tmp_taper
        end subroutine feature_iso_devel
        
        subroutine feature_iso_devel_deriv(set_type,conf,atm,neigh_idx,ft_idx,deriv_vec,deriv_idx)
            use propagate, only : logistic,logistic_deriv
            
            implicit none

            integer,intent(in) :: set_type,conf,atm,neigh_idx,ft_idx,deriv_idx
            real(8),intent(inout) :: deriv_vec(1:3)

            integer :: lim1,lim2,ii
            real(8) :: tmp2,dr_scl,dr_vec(1:3)
            real(8) :: rcuts(1:2),mean,const,r_taper,std
            real(8) :: xtilde,sig,sig_prime,fs
            real(8) :: tap,tap_deriv,r_nl(1:3)

            mean = feature_params%info(ft_idx)%devel(1)
            const = feature_params%info(ft_idx)%devel(2)
            std = feature_params%info(ft_idx)%devel(3)
            fs = feature_params%info(ft_idx)%fs

            !* x-> const*(x-mean) up to x-mean = 2*std

            rcuts(1) = feature_params%info(ft_idx)%rcut
            rcuts(2) = mean + 2.0d0*std 

            r_taper = minval(rcuts)
            
            if (neigh_idx.eq.0) then
                lim1 = 1
                lim2 = set_neigh_info(conf)%twobody(atm)%n
                tmp2 = -1.0d0       !* sign for drij/d r_central
                if (calculate_property("stress")) then
                    r_nl = set_neigh_info(conf)%twobody(atm)%r_nl_atom
                end if
            else
                lim1 = neigh_idx
                lim2 = neigh_idx    
                tmp2 = 1.0d0        !* sign for drij/d r_neighbour
                if (calculate_property("stress")) then
                    r_nl = set_neigh_info(conf)%twobody(atm)%r_nl_neigh(:,lim1)
                end if
            end if


            !* derivative wrt. central atom itself
            do ii=lim1,lim2,1
                if (atm.eq.set_neigh_info(conf)%twobody(atm)%idx(ii)) then
                    ! dr_vec =  d (r_i + const - r_i ) / d r_i = 0
                    cycle
                end if
                
                !* atom-atom distance
                dr_scl = set_neigh_info(conf)%twobody(atm)%dr(ii)

                if (dr_scl.gt.r_taper) then
                    cycle
                end if

                !* (r_neighbour - r_centralatom)/dr_scl
                dr_vec(:) = set_neigh_info(conf)%twobody(atm)%drdri(:,ii)
                
                !* tapering
                tap = taper_1(dr_scl,r_taper,fs)
                tap_deriv = taper_deriv_1(dr_scl,r_taper,fs)

                sig = logistic(xtilde)
                sig_prime = logistic_deriv(xtilde)*const


                deriv_vec(:) = deriv_vec(:) + (sig*tap_deriv + sig_prime*tap)*dr_vec(:)*tmp2
                
                if (calculate_property("stress")) then
                    call append_stress_contribution((sig*tap_deriv+sig_prime*tap)*tmp2*dr_vec,&
                            &r_nl,set_type,conf,atm,ft_idx,deriv_idx)
                end if
            end do
        end subroutine feature_iso_devel_deriv

        subroutine feature_normal_threebody(set_type,conf,atm,ft_idx,bond_idx)
            implicit none

            !* args
            integer,intent(in) :: atm,ft_idx,bond_idx,set_type,conf

            !* scratch
            real(8) :: prec(1:3,1:3)
            real(8) :: mean(1:3),fs,rcut,za,zb
            real(8) :: tmp_atmz,tmp_taper,x1(1:3),x2(1:3)
            real(8) :: drij,drik,drjk,cos_angle,sqrt_det
            real(8) :: zatom,zneigh1,zneigh2

            !* feature parameters
            rcut     = feature_params%info(ft_idx)%rcut
            prec     = feature_params%info(ft_idx)%prec
            mean     = feature_params%info(ft_idx)%mean
            fs       = feature_params%info(ft_idx)%fs
            za       = feature_params%info(ft_idx)%za
            zb       = feature_params%info(ft_idx)%zb
            sqrt_det = feature_params%info(ft_idx)%sqrt_det

            !* atom-atom distances
            drij = set_neigh_info(conf)%threebody(atm)%dr(1,bond_idx)
            drik = set_neigh_info(conf)%threebody(atm)%dr(2,bond_idx)
            drjk = set_neigh_info(conf)%threebody(atm)%dr(3,bond_idx)
            
            if ( (drij.gt.rcut).or.(drik.gt.rcut).or.(drjk.gt.rcut) ) then
                return
            end if

            cos_angle = set_neigh_info(conf)%threebody(atm)%cos_ang(bond_idx)

            !* must permute atom order to retain invariance
            x1(1) = drij
            x1(2) = drik
            x1(3) = cos_angle
            x2(1) = drik
            x2(2) = drij
            x2(3) = cos_angle


            !* taper term
            if (speedup_applies("threebody_rcut")) then
                tmp_taper = product(set_neigh_info(conf)%threebody(atm)%dr_taper(1:3,bond_idx))
            else
                tmp_taper = taper_1(drij,rcut,fs)*taper_1(drik,rcut,fs)*taper_1(drjk,rcut,fs)
            end if

            if (speedup_applies("single_element")) then
                tmp_atmz = atomic_weighting(-1.0d0,-1.0d0,-1.0d0,ft_idx)
            else           
                zatom = set_neigh_info(conf)%threebody(atm)%z_atom
                zneigh1 = set_neigh_info(conf)%threebody(atm)%z(1,bond_idx)
                zneigh2 = set_neigh_info(conf)%threebody(atm)%z(2,bond_idx)
            
                tmp_atmz = atomic_weighting(zatom,zneigh1,zneigh2,ft_idx)
            end if
            
            data_sets(set_type)%configs(conf)%x(ft_idx+1,atm) = &
                    &data_sets(set_type)%configs(conf)%x(ft_idx+1,atm)&
                    + (func_normal(x1,mean,prec) + func_normal(x2,mean,prec)) * &
                    &tmp_atmz*tmp_taper
        end subroutine feature_normal_threebody
        
        subroutine feature_normal_threebody_deriv(set_type,conf,atm,ft_idx,bond_idx,idx_to_contrib) 
            implicit none

            !* args
            integer,intent(in) :: set_type,conf,atm,ft_idx,bond_idx
            integer,intent(in) :: idx_to_contrib(1:2)
            
            !* scratch
            real(8) :: prec(1:3,1:3),sqrt_det
            real(8) :: mean(1:3),fs,rcut,za,zb
            real(8) :: x(1:3,1:2),drjkdrz(1:3)
            real(8) :: drij,drik,drjk,cos_angle
            real(8) :: tmp_feat(1:2),tmp_vec(1:3,1:2)
            real(8) :: tmp_deriv1(1:3),tmp_deriv2(1:3)
            real(8) :: dxdr1(1:3,1:3),dxdr2(1:3,1:3),r_nl(1:3),dxdr(1:3)
            real(8) :: dcosdrz(1:3),drijdrz(1:3),drikdrz(1:3),tap_ij,tap_ik
            real(8) :: tap_ij_deriv,tap_ik_deriv,tmpz,tap_jk,tap_jk_deriv
            real(8) :: zatom,zneigh1,zneigh2
            integer :: deriv_idx,zz

            !* feature parameters
            rcut     = feature_params%info(ft_idx)%rcut
            prec     = feature_params%info(ft_idx)%prec
            mean     = feature_params%info(ft_idx)%mean
            fs       = feature_params%info(ft_idx)%fs
            za       = feature_params%info(ft_idx)%za
            zb       = feature_params%info(ft_idx)%zb
            sqrt_det = feature_params%info(ft_idx)%sqrt_det

            !* atom-atom distances
            drij = set_neigh_info(conf)%threebody(atm)%dr(1,bond_idx)
            drik = set_neigh_info(conf)%threebody(atm)%dr(2,bond_idx)
            drjk = set_neigh_info(conf)%threebody(atm)%dr(3,bond_idx)
            
            if ( (drij.gt.rcut).or.(drik.gt.rcut).or.(drjk.gt.rcut) ) then
                return
            end if

            cos_angle = set_neigh_info(conf)%threebody(atm)%cos_ang(bond_idx)

            !* must permuate atom ordering to retain invariance
            x(1,1) = drij
            x(2,1) = drik
            x(1,2) = drik
            x(2,2) = drij 
            x(3,:) = cos_angle

            !* tapering
            if (speedup_applies("threebody_rcut")) then
                tap_ij = set_neigh_info(conf)%threebody(atm)%dr_taper(1,bond_idx)
                tap_ik = set_neigh_info(conf)%threebody(atm)%dr_taper(2,bond_idx)
                tap_jk = set_neigh_info(conf)%threebody(atm)%dr_taper(3,bond_idx)
                tap_ij_deriv = set_neigh_info(conf)%threebody(atm)%dr_taper_deriv(1,bond_idx)
                tap_ik_deriv = set_neigh_info(conf)%threebody(atm)%dr_taper_deriv(2,bond_idx)
                tap_jk_deriv = set_neigh_info(conf)%threebody(atm)%dr_taper_deriv(3,bond_idx)
            else
                tap_ij = taper_1(drij,rcut,fs)
                tap_ik = taper_1(drik,rcut,fs)
                tap_jk = taper_1(drjk,rcut,fs)
                tap_ij_deriv = taper_deriv_1(drij,rcut,fs)
                tap_ik_deriv = taper_deriv_1(drik,rcut,fs)
                tap_jk_deriv = taper_deriv_1(drjk,rcut,fs)
            end if

            if (speedup_applies("single_element")) then
                tmpz = atomic_weighting(-1.0d0,-1.0d0,-1.0d0,ft_idx)
            else           
                zatom = set_neigh_info(conf)%threebody(atm)%z_atom
                zneigh1 = set_neigh_info(conf)%threebody(atm)%z(1,bond_idx)
                zneigh2 = set_neigh_info(conf)%threebody(atm)%z(2,bond_idx)
            
                tmpz = atomic_weighting(zatom,zneigh1,zneigh2,ft_idx)
            end if

            tmp_feat(1) = func_normal(x(:,1),mean,prec)
            tmp_feat(2) = func_normal(x(:,2),mean,prec)

            do zz=1,2
                !* prec * (x - mean)
                call dsymv('u',3,1.0d0,prec,3,x(:,zz)-mean,1,0.0d0,tmp_vec(:,zz),1)
            end do    
        
            
            ! 1=jj , 2=kk, 3=ii
            do zz=1,3,1
                ! map atom id to portion of mem for derivative
                if (zz.lt.3) then
                    deriv_idx = idx_to_contrib(zz) 
                else
                    deriv_idx = 1
                end if
                
                !* derivatives wrt r_zz
                dcosdrz =  set_neigh_info(conf)%threebody(atm)%dcos_dr(:,zz,bond_idx)
                
                if (zz.eq.1) then
                    ! zz=jj
                    drijdrz =  set_neigh_info(conf)%threebody(atm)%drdri(:,1,bond_idx)
                    drikdrz =  set_neigh_info(conf)%threebody(atm)%drdri(:,4,bond_idx)
                    drjkdrz = -set_neigh_info(conf)%threebody(atm)%drdri(:,5,bond_idx)
                else if (zz.eq.2) then
                    ! zz=kk
                    drijdrz =  set_neigh_info(conf)%threebody(atm)%drdri(:,2,bond_idx)
                    drikdrz =  set_neigh_info(conf)%threebody(atm)%drdri(:,3,bond_idx)
                    drjkdrz =  set_neigh_info(conf)%threebody(atm)%drdri(:,5,bond_idx)
                else if (zz.eq.3) then
                    ! zz=ii
                    drijdrz = -set_neigh_info(conf)%threebody(atm)%drdri(:,1,bond_idx)
                    drikdrz = -set_neigh_info(conf)%threebody(atm)%drdri(:,3,bond_idx)
                    drjkdrz =  set_neigh_info(conf)%threebody(atm)%drdri(:,6,bond_idx)
                end if
                if (calculate_property("stress")) then
                    r_nl = set_neigh_info(conf)%threebody(atm)%r_nl(:,zz,bond_idx)    
                end if

                !* dx_{ijk} / dr_z |_ab = d x_{ijk}|_b / d r_z|_a
                dxdr1(:,1) = drijdrz
                dxdr1(:,2) = drikdrz
                dxdr1(:,3) = dcosdrz
                dxdr2(:,1) = drikdrz
                dxdr2(:,2) = drijdrz
                dxdr2(:,3) = dcosdrz

                !* tmp_deriv = dxdr * tmp_vec
                call dgemv('n',3,3,1.0d0,dxdr1,3,tmp_vec(:,1),1,0.0d0,tmp_deriv1,1)
                call dgemv('n',3,3,1.0d0,dxdr2,3,tmp_vec(:,2),1,0.0d0,tmp_deriv2,1)
                        
                dxdr = sum(tmp_feat)*(tap_ij*tap_jk*tap_ik_deriv*drikdrz + &
                        &tap_ik*tap_jk*tap_ij_deriv*drijdrz)*tmpz - &
                        &tap_ij*tap_ik*tap_jk*(tmp_feat(1)*tmp_deriv1+tmp_feat(2)*tmp_deriv2)*tmpz &
                        + tap_ij*tap_ik*tmpz*sum(tmp_feat)*tap_jk_deriv*drjkdrz !* this line is new

                data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(:,deriv_idx) = &
                        &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(:,deriv_idx) + & 
                        &dxdr
                
                if (calculate_property("stress")) then
                    call append_stress_contribution(dxdr,r_nl,set_type,conf,atm,ft_idx,deriv_idx)
                end if
                 
            end do
            
        end subroutine feature_normal_threebody_deriv
        
        subroutine feature_fourier_b2(conf,atm,neigh_idx,ft_idx,current_val)
            implicit none

            integer,intent(in) :: conf,atm,neigh_idx,ft_idx
            real(8),intent(inout) :: current_val

            !* scratch
            real(8) :: dr,tmp1,tmp2,tmp3,za,zb,rcut,fs
            real(8) :: phi(1:size(feature_params%info(ft_idx)%linear_w))
            real(8) :: zatom,zneigh,kk_dble
            integer :: kk,num_weights,fourier_terms
           
            !* atom-neigh_idx distance 
            dr = set_neigh_info(conf)%twobody(atm)%dr(neigh_idx)

            !* symmetry function params
            za   = feature_params%info(ft_idx)%za
            zb   = feature_params%info(ft_idx)%zb
            rcut = feature_params%info(ft_idx)%rcut
            fs   = feature_params%info(ft_idx)%fs

            !* number of weights (1 weight per sin/cos)
            num_weights = size(feature_params%info(ft_idx)%linear_w)

            !* number of sin + cos terms
            fourier_terms = int(num_weights/2)

            !* 2 pi / rcut
            tmp1 = dr * 6.28318530718 / rcut

            !* tapering
            if (speedup_applies("twobody_rcut")) then
                tmp2 = set_neigh_info(conf)%twobody(atm)%dr_taper(neigh_idx)
            else
                tmp2 = taper_1(dr,rcut,fs)
            end if
           
            if (speedup_applies("single_element")) then
                tmp3 = atomic_weighting(-1.0d0,-1.0d0,-1.0d0,ft_idx)
            else
                zatom = set_neigh_info(conf)%twobody(atm)%z_atom
                zneigh = set_neigh_info(conf)%twobody(atm)%z(neigh_idx)
                
                tmp3 = atomic_weighting(zatom,zneigh,-1.0d0,ft_idx)
            end if
       
            ! we don't care about constant offset, discard 0th contribution
            do kk=1,fourier_terms,1
                kk_dble = dble(kk)

                !* w = (w_sin , w_cos) -> phi = (phi_sin, phi_cos)
                phi(kk) = sin(kk_dble*tmp1)
                phi(kk+fourier_terms) = cos(kk_dble*tmp1)
            end do !*end loop over linear model weights
        
            current_val = current_val + &
                    &ddot(num_weights,feature_params%info(ft_idx)%linear_w,1,phi,1)*tmp2*tmp3

        end subroutine feature_fourier_b2
        
        subroutine feature_fourier_b2_deriv(set_type,conf,atm,neigh_idx,ft_idx,deriv_vec,deriv_idx)
            implicit none

            integer,intent(in) :: set_type,conf,atm,neigh_idx,ft_idx,deriv_idx
            real(8),intent(inout) :: deriv_vec(1:3)

            !* scratch
            real(8) :: dr_scl,dr_vec(1:3),tap_deriv,tap,tmp1,tmp2
            real(8) :: fs,rcut,tmpz,tmp3,r_nl(1:3)
            real(8) :: za,zb,kk_dble,phi(1:size(feature_params%info(ft_idx)%linear_w))
            real(8) :: zatom,zneigh,tmp4,tmps,tmpc
            integer :: ii,kk,lim1,lim2,num_weights,fourier_terms

            !* symmetry function params
            za   = feature_params%info(ft_idx)%za
            zb   = feature_params%info(ft_idx)%zb
            fs   = feature_params%info(ft_idx)%fs
            rcut = feature_params%info(ft_idx)%rcut

            !* number of functions in linear model (no bias)
            num_weights = size(feature_params%info(ft_idx)%linear_w)
            
            !* number of sin + cos terms
            fourier_terms = int(num_weights/2)

            !* 2 pi / rcut
            tmp1 = 6.28318530718d0 / rcut
            
            if (neigh_idx.eq.0) then
                lim1 = 1
                lim2 = set_neigh_info(conf)%twobody(atm)%n
                tmp2 = -1.0d0       !* sign for drij/d r_central
                if (calculate_property("stress")) then
                    r_nl = set_neigh_info(conf)%twobody(atm)%r_nl_atom
                end if
            else
                lim1 = neigh_idx
                lim2 = neigh_idx    
                tmp2 = 1.0d0        !* sign for drij/d r_neighbour
                if (calculate_property("stress")) then
                    r_nl = set_neigh_info(conf)%twobody(atm)%r_nl_neigh(:,lim1)
                end if
            end if
            
            if (speedup_applies("single_element")) then
                tmpz = atomic_weighting(-1.0d0,-1.0d0,-1.0d0,ft_idx)
            end if


            !* derivative wrt. central atom itself
            do ii=lim1,lim2,1
                ! NO SELF INTERACTION
                if (atm.eq.set_neigh_info(conf)%twobody(atm)%idx(ii)) then
                    ! dr_vec =  d (r_i + const - r_i ) / d r_i = 0
                    cycle
                end if
                
                !* atom-atom distance
                dr_scl = set_neigh_info(conf)%twobody(atm)%dr(ii)

                !* (r_neighbour - r_centralatom)/dr_scl
                dr_vec(:) = set_neigh_info(conf)%twobody(atm)%drdri(:,ii)
                
                !* tapering
                if (speedup_applies("twobody_rcut")) then
                    tap = set_neigh_info(conf)%twobody(atm)%dr_taper(ii)
                    tap_deriv = set_neigh_info(conf)%twobody(atm)%dr_taper_deriv(ii)
                else
                    tap = taper_1(dr_scl,rcut,fs)
                    tap_deriv = taper_deriv_1(dr_scl,rcut,fs)
                end if

                !* atomic number
                if (.not.speedup_applies("single_element")) then
                    zatom = set_neigh_info(conf)%twobody(atm)%z_atom
                    zneigh = set_neigh_info(conf)%twobody(atm)%z(ii)
                    tmpz = atomic_weighting(zatom,zneigh,-1.0d0,ft_idx)
                end if

                do kk=1,fourier_terms,1
                    kk_dble = dble(kk)
                    tmp4 = kk_dble*tmp1*dr_scl
                    tmps = sin(tmp4)
                    tmpc = cos(tmp4)
                   
                    phi(kk) = tap_deriv*tmps + tap*tmp1*kk_dble*tmpc
                    phi(kk+fourier_terms) = tap_deriv*tmpc - tap*tmp1*kk_dble*tmps
                end do !* end loop over linear model weights

                !* Gamma' \sum_k w_k cos(2 pi k dr/rcut)  + 
                !* Gamma \sum_k w_k 2*pi*k/rcut*sin(2 pi k dr/rcut)
                tmp3 = ddot(num_weights,feature_params%info(ft_idx)%linear_w,1,phi,1) 

                deriv_vec(:) = deriv_vec(:) + dr_vec(:)*tmp3*tmp2*tmpz
                
                if (calculate_property("stress")) then
                    call append_stress_contribution(dr_vec*tmp3*tmp2*tmpz,&
                            &r_nl,set_type,conf,atm,ft_idx,deriv_idx)
                end if
            end do
        end subroutine feature_fourier_b2_deriv

        real(8) function atomic_weighting(zlocal,zneigh1,zneigh2,ft)
            implicit none

            !* args
            real(8),intent(in) :: zlocal,zneigh1,zneigh2
            integer,intent(in) :: ft

            !* scratch
            real(8) :: za,zb
            real(8) :: val

            !* if twobody, zneigh2 = -1.0d0

            if (speedup_applies("single_element")) then
                val = feature_params%info(ft)%z_single_element
            else
                za = feature_params%info(ft)%za
                zb = feature_params%info(ft)%zb

                if (feature_IsTwoBody(feature_params%info(ft)%ftype)) then
                    val = atomic_weighting_twobody_calc(zlocal,zneigh1,za,zb)
                else
                    val = atomic_weighting_threebody_calc(zlocal,zneigh1,zneigh2,za,zb)
                end if
            end if
            atomic_weighting = val
        end function atomic_weighting

        real(8) function atomic_weighting_twobody_calc(zlocal,zneigh,za,zb)
            implicit none

            real(8),intent(in) :: zlocal,zneigh,za,zb

            atomic_weighting_twobody_calc = ((zlocal+1.0d0)**za) * ((zneigh+1.0d0)**zb)
        end function atomic_weighting_twobody_calc
        
        real(8) function atomic_weighting_threebody_calc(zlocal,zneigh1,zneigh2,za,zb)
            implicit none

            real(8),intent(in) :: zlocal,zneigh1,zneigh2,za,zb

            real(8) :: tmp

            tmp = ((zlocal+1.0d0)**za) * ( ((zneigh1+1.0d0)*(zneigh2+1.0d0))**zb )
            atomic_weighting_threebody_calc = tmp
        end function atomic_weighting_threebody_calc
                    
                    
        subroutine append_stress_contribution(dxdr_cont,r_nl,&
        &set_type,conf,atm,ft,deriv_idx)
            implicit none

            real(8),intent(in) :: dxdr_cont(1:3),r_nl(1:3)
            integer,intent(in) :: set_type,conf,atm,ft,deriv_idx

            !* scratch
            integer :: xx,yy

            if (running_unittest) then
                !* dont symmetrize stress matrix, can check diagonals as test
                do xx=1,3
                    do yy=1,3
                        data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%stress(xx,yy,deriv_idx)=&
                        &data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%stress(xx,yy,deriv_idx)-&
                                &dxdr_cont(xx)*r_nl(yy)
                    end do
                end do
            else
                do xx=1,3
                    data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%stress(xx,xx,deriv_idx)=&
                    &data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%stress(xx,xx,deriv_idx)-&
                            &dxdr_cont(xx)*r_nl(xx)
                end do
                data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%stress(2,1,deriv_idx)=&
                &data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%stress(2,1,deriv_idx)-&
                        &0.5d0*(dxdr_cont(1)*r_nl(2) + dxdr_cont(2)*r_nl(1))
                
                data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%stress(3,1,deriv_idx)=&
                &data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%stress(3,1,deriv_idx)-&
                        &0.5d0*(dxdr_cont(1)*r_nl(3) + dxdr_cont(3)*r_nl(1))
                
                data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%stress(1,2,deriv_idx)=&
                &data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%stress(1,2,deriv_idx)-&
                        &0.5d0*(dxdr_cont(2)*r_nl(1) + dxdr_cont(1)*r_nl(2))
                
                data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%stress(3,2,deriv_idx)=&
                &data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%stress(3,2,deriv_idx)-&
                        &0.5d0*(dxdr_cont(2)*r_nl(3) + dxdr_cont(3)*r_nl(2))
                
                data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%stress(1,3,deriv_idx)=&
                &data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%stress(1,3,deriv_idx)-&
                        &0.5d0*(dxdr_cont(1)*r_nl(3) + dxdr_cont(3)*r_nl(1))
                
                data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%stress(2,3,deriv_idx)=&
                &data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%stress(2,3,deriv_idx)-&
                        &0.5d0*(dxdr_cont(2)*r_nl(3) + dxdr_cont(3)*r_nl(2))
            end if

        end subroutine append_stress_contribution
        
        subroutine direct_stress_contribution(dxdr,r_nl,ft,atm,stress_tensor)
            implicit none

            !* args
            real(8),intent(in) :: dxdr(1:3),r_nl(1:3)
            integer,intent(in) :: ft,atm
            real(8),intent(inout) :: stress_tensor(1:3,1:3)
                
                
            integer :: xx,yy
            
            do xx=1,3
                do yy=1,3
                    stress_tensor(xx,yy) = stress_tensor(xx,yy) + dxdr(xx)*r_nl(yy)*dydx(ft,atm)
                end do
            end do
        end subroutine direct_stress_contribution

        subroutine direct_force_contribution(dxdr,ft,atm,fxx,fyy,fzz)
            implicit none

            real(8),intent(in) :: dxdr(1:3)
            real(8),intent(inout) :: fxx(:),fyy(:),fzz(:)
            integer,intent(in) :: ft,atm

            !* for compatibility with DLPOLY
            fxx(atm) = fxx(atm) + dxdr(1)*dydx(ft,atm) 
            fyy(atm) = fyy(atm) + dxdr(2)*dydx(ft,atm) 
            fzz(atm) = fzz(atm) + dxdr(3)*dydx(ft,atm) 
        end subroutine direct_force_contribution

        subroutine experimental_feature_calc(set_type,conf,updating_features)
            implicit none

            !* args
            integer,intent(in) :: set_type,conf
            logical,intent(in) :: updating_features
        
            !* scratch
            integer :: atm,ft
            
            if (updating_features) then
                updating_net_weights_only = .false.
            else
                !* remove redundant neighbour info
                updating_net_weights_only = .true.
            end if
            
            !* two body contributions
            if (twobody_features_present()) then
                do atm=1,data_sets(set_type)%configs(conf)%n,1
                    call twobody_atom_contribution(set_type,conf,atm)
                end do !* end loop over central 2body atoms
            end if
            !* three body contributions
            if (threebody_features_present()) then
                do atm=1,data_sets(set_type)%configs(conf)%n,1
                    call threebody_atom_contribution(set_type,conf,atm)
                end do !* end loop over central 3body atoms
            end if
            !* atomic number feature
            do ft=1,feature_params%num_features,1
                if (feature_params%info(ft)%ftype.eq.featureID_StringToInt("atomic_number")) then
                    do atm=1,data_sets(set_type)%configs(conf)%n,1
                        call feature_atomicnumber_1(set_type,conf,atm,ft) 
                    end do !* end loop over atoms 
                end if
            end do
        end subroutine experimental_feature_calc

        subroutine twobody_atom_contribution(set_type,conf,atm)
            use lookup
            
            implicit none

            !* args
            integer,intent(in) :: set_type,conf,atm

            !* scratch
            integer :: neigh,ft,ftype,cntr,arg,zz,deriv_idx
            integer :: num_weights,ww,deriv_atm
            integer :: contrib_atms(1:data_sets(set_type)%configs(conf)%n)
            integer :: idx_to_contrib(1:set_neigh_info(conf)%twobody(atm)%n)
            real(8) :: rcut,dr,r_nl(1:3),r_nl_central(1:3),dr_vec(1:3)
            real(8) :: rcut_ft,taper,taper_deriv,tmpz,feat_val
            real(8) :: feat_deriv,feat_deriv_vec(1:3),ww_dble
            real(8) :: eta,rs,prec,mean,phi(1:1000),const,scl
            real(8) :: fs,za,zb
            logical :: nonzero_derivative
            
            if (set_neigh_info(conf)%twobody(atm)%n.eq.0) then
                !* no twobody interactions for this central atom
                do ft=1,feature_params%num_features
                    if (.not.feature_IsTwoBody(feature_params%info(ft)%ftype)) then
                        cycle
                    end if
                    data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%n = 0
                end do
                return
            end if

            !* max rcut of any 2body feature
            rcut = maxrcut(1)

            if (calculate_property("forces").and.(.not.calculation_type("single_point"))) then
                !* local coordinate of central atom
                r_nl_central = set_neigh_info(conf)%twobody(atm)%r_nl_atom

                !* for full periodic boundaries, need to identify same local atoms
                contrib_atms(1) = atm
                cntr = 1
                do neigh=1,set_neigh_info(conf)%twobody(atm)%n,1
                    if ( int_in_intarray(set_neigh_info(conf)%twobody(atm)%idx(neigh),&
                    &contrib_atms(1:cntr),arg) ) then
                        !* Local atom already in list, note corresponding idx in contrib_atms
                        idx_to_contrib(neigh) = arg
                        cycle
                    else if (set_neigh_info(conf)%twobody(atm)%dr(neigh).le.rcut) then
                        cntr = cntr + 1
                        !* note this local atom contributes to this feature for atom
                        contrib_atms(cntr) = set_neigh_info(conf)%twobody(atm)%idx(neigh)
                        idx_to_contrib(neigh) = cntr
                    else
                        !* beyond cut off
                        idx_to_contrib(neigh) = -1     ! NULL value
                    end if
                end do !* end loop over neighbours to atm

                do ft=1,feature_params%num_features,1
                    if (.not.feature_IsTwoBody(feature_params%info(ft)%ftype)) then
                        cycle
                    end if

                    !* mem allocation
                    if (allocated(data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%idx)) then
                        deallocate(data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%idx)
                    end if
                    if (allocated(data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%vec)) then
                        deallocate(data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%vec)
                    end if
                    if (allocated(data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%stress)) then
                        deallocate(data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%stress)
                    end if

                    allocate(data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%idx(1:cntr))
                    allocate(data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%vec(1:3,1:cntr))
                    if (calculate_property("stress")) then
                        allocate(data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%&
                                &stress(1:3,1:3,1:cntr))
                    end if

                    !* even though some elements may be zero, set num neighours to max possible
                    data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%n = cntr
                    data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%idx(:) = &
                            &contrib_atms(1:cntr) !* can us LAPACK here

                    data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%vec = 0.0d0
                    if (calculate_property("stress")) then
                        data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%stress = 0.0d0
                    end if

                end do !* end loop over features
            end if !* forces

            if (.not.calculation_type("single_point")) then
                do ft=1,feature_params%num_features
                    if (feature_IsTwoBody(feature_params%info(ft)%ftype)) then
                        data_sets(set_type)%configs(conf)%x(ft+1,atm) = 0.0d0
                    end if
                end do
            end if

            do neigh=1,set_neigh_info(conf)%twobody(atm)%n,1
                if (atm.eq.set_neigh_info(conf)%twobody(atm)%idx(neigh)) then
                    nonzero_derivative = .false.
                else
                    nonzero_derivative = .true.
                end if

                !* atm-neigh distance
                dr = set_neigh_info(conf)%twobody(atm)%dr(neigh)
                
                do ft=1,feature_params%num_features
                    ftype = feature_params%info(ft)%ftype
                    
                    if (.not.feature_params%info(ft)%is_twobody) then
                        cycle
                    end if

                    rcut_ft = feature_params%info(ft)%rcut
                    if (dr.gt.rcut_ft) then
                        !* feature rcuts can be different
                        cycle
                    end if
                    
                    za   = feature_params%info(ft)%za
                    zb   = feature_params%info(ft)%zb
                    fs   = feature_params%info(ft)%fs
                    rs   = feature_params%info(ft)%rs
                    eta  = feature_params%info(ft)%eta
                    scl  = feature_params%info(ft)%scl_cnst
                    if (ftype.eq.featureID_StringToInt("acsf_normal-b2")) then
                        prec = feature_params%info(ft)%prec(1,1)
                        mean = feature_params%info(ft)%mean(1)
                    else if (ftype.eq.featureID_StringToInt("acsf_fourier-b2")) then
                        num_weights = size(feature_params%info(ft)%linear_w)
                    end if

                    if (speedup_applies("twobody_rcut")) then
                        taper = set_neigh_info(conf)%twobody(atm)%dr_taper(neigh)
                        if (calculate_property("forces")) then
                            taper_deriv = set_neigh_info(conf)%twobody(atm)%dr_taper_deriv(neigh)
                        end if
                    else
                        taper = taper_1(dr,rcut_ft,fs)
                        if (calculate_property("forces")) then
                            taper_deriv = taper_deriv_1(dr,rcut_ft,fs)
                        end if
                    end if

                    !* atomic number weighting
                    tmpz = (set_neigh_info(conf)%twobody(atm)%z_atom+1.0d0)**za * &
                            &(set_neigh_info(conf)%twobody(atm)%z(neigh)+1.0d0)**zb
                    
                    !* 0th derivative contribution
                    !if (.not.calculation_type("single_point")) then
                   
                        if (ftype.eq.featureID_StringToInt("acsf_behler-g1")) then 
                            feat_val = taper*tmpz*scl 
                        else if (ftype.eq.featureID_StringToInt("acsf_behler-g2")) then
                            if (speedup_applies("lookup_tables")) then
                                feat_val = access_lookup(dr,map_to_tbl_idx(1,ft))*tmpz
                            else
                                feat_val = exp(-eta*(dr-rs)**2)*taper*tmpz*scl 
                            end if
                        else if (ftype.eq.featureID_StringToInt("acsf_normal-b2")) then
                            if (speedup_applies("lookup_tables")) then
                                feat_val = access_lookup(dr,map_to_tbl_idx(1,ft))*tmpz
                            else
                                feat_val = exp(-0.5d0*prec*((dr-mean)**2))*taper*tmpz*scl
                            end if
                        else if (ftype.eq.featureID_StringToInt("acsf_fourier-b2")) then
                            if (speedup_applies("lookup_tables")) then
                                feat_val = access_lookup(dr,map_to_tbl_idx(1,ft))*tmpz
                            else
                                const = dr * 6.28318530718 / rcut_ft
                                do ww=1,num_weights,1
                                    phi(ww) = sin(dble(ww)*const)
                                end do
                                feat_val = ddot(num_weights,feature_params%info(ft)%linear_w,&
                                        &1,phi,1)*taper*tmpz*scl
                            end if 
                        else                        
                            call error("twobody_atom_contribution","Implementation error")
                        end if

                        data_sets(set_type)%configs(conf)%x(ft+1,atm) = &
                                &data_sets(set_type)%configs(conf)%x(ft+1,atm) + feat_val
                    !end if !* for single_point, x are computed in twobody_info routine

                    if (calculate_property("forces").and.nonzero_derivative) then 
                        do zz=1,2
                            !* zz=1, derivative wrt neigh, zz=2 : wrt. central atom
                            if (zz.eq.1) then
                                deriv_idx = idx_to_contrib(neigh)
                                deriv_atm = set_neigh_info(conf)%twobody(atm)%idx(neigh)
                                dr_vec = set_neigh_info(conf)%twobody(atm)%drdri(:,neigh)
                                if (calculate_property("stress")) then
                                    r_nl = set_neigh_info(conf)%twobody(atm)%r_nl_neigh(:,neigh)
                                end if
                            else
                                deriv_idx = 1
                                deriv_atm = atm 
                                dr_vec = -set_neigh_info(conf)%twobody(atm)%drdri(:,neigh)
                                if (calculate_property("stress")) then
                                    r_nl = r_nl_central
                                end if
                            end if

                            if (ftype.eq.featureID_StringToInt("acsf_behler-g1")) then
                                feat_deriv = taper_deriv*tmpz*scl
                            else if (ftype.eq.featureID_StringToInt("acsf_behler-g2")) then
                                if (speedup_applies("lookup_tables")) then
                                    feat_deriv = access_lookup(dr,map_to_tbl_idx(2,ft))*tmpz
                                else
                                    feat_deriv = exp(-eta*(dr-rs)**2)  *  (taper_deriv - &
                                            &2.0d0*eta*(dr-rs)*taper) * tmpz * scl
                                end if
                            else if (ftype.eq.featureID_StringToInt("acsf_normal-b2")) then
                                if (speedup_applies("lookup_tables")) then
                                    feat_deriv = access_lookup(dr,map_to_tbl_idx(2,ft))*tmpz
                                else
                                    feat_deriv = exp(-0.5d0*prec*(dr-mean)**2)*(taper_deriv - & 
                                            &prec*(dr-mean)*taper)*tmpz*scl
                                end if
                            else if (ftype.eq.featureID_StringToInt("acsf_fourier-b2")) then
                                if (speedup_applies("lookup_tables")) then
                                    feat_deriv = access_lookup(dr,map_to_tbl_idx(2,ft))*tmpz
                                else
                                    const = 6.28318530718 / rcut_ft
                                    do ww=1,num_weights,1
                                        ww_dble = dble(ww)

                                        phi(ww) = taper_deriv*sin(ww_dble*const*dr) + &
                                                &taper*const*ww_dble*cos(ww_dble*const*dr)
                                    end do
                                    
                                    feat_deriv = ddot(num_weights,&
                                            &feature_params%info(ft)%linear_w,1,phi,1)*tmpz*scl
                                end if
                            end if

                            !* dx / dr_derividx
                            feat_deriv_vec = dr_vec*feat_deriv

                            !* force contribution
                            if (calculation_type("single_point")) then
                                !* dydx is known
                                call direct_force_contribution(feat_deriv_vec,ft,deriv_atm,&
                                        &data_sets(set_type)%configs(conf)%current_fi(1,:),&
                                        &data_sets(set_type)%configs(conf)%current_fi(2,:),&
                                        &data_sets(set_type)%configs(conf)%current_fi(3,:))
                            else
                                !* dydx is unknown
                                data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%vec(1:3,&
                                        &deriv_idx) = data_sets(set_type)%configs(conf)%&
                                        &x_deriv(ft,atm)%vec(1:3,deriv_idx) + feat_deriv_vec
                            end if

                            if (calculate_property("stress")) then
                                if (calculation_type("single_point")) then
                                    !* dydx is known
                                    call direct_stress_contribution(feat_deriv_vec,r_nl,ft,&
                                            &deriv_atm,&
                                            &data_sets(set_type)%configs(conf)%current_stress)
                                else
                                    !* dydx is unknown
                                    call append_stress_contribution(feat_deriv_vec,r_nl,set_type,&
                                        &conf,atm,ft,deriv_idx)
                                end if
                            end if
                        end do !* end loop over zz
                    end if !* forces
                end do !* end loop over ft features
            end do !* end loop over neighbours to central atom
        end subroutine twobody_atom_contribution

        subroutine threebody_atom_contribution(set_type,conf,atm)
            use lookup
            
            implicit none

            !* args
            integer,intent(in) :: set_type,conf,atm

            !* scratch
            integer :: ft,idx_map(1:2,1:set_neigh_info(conf)%threebody(atm)%n)
            integer :: contrib_atms(1:data_sets(set_type)%configs(conf)%n)
            integer :: cntr,neigh,ii,bond,zz,ftype,deriv_idx,arg
            real(8) :: x1(1:3),x2(1:3),prec(1:3,1:3),mean(1:3)
            real(8) :: dcosdrz(1:3),drijdrz(1:3),drikdrz(1:3),tmp3
            real(8) :: drjkdrz(1:3),tmp1,tmp2,dxdr(1:3),tmp_vec1(1:3),tmp_vec2(1:3)
            real(8) :: norm_tmp1(1:3,1:3),norm_tmp2(1:3,1:3),tmp_deriv1(1:3),tmp_deriv2(1:3)
            real(8) :: za,zb,eta,fs,xi,lambda,tap_ij,tap_ik,tap_jk,tap_ij_deriv
            real(8) :: tap_ik_deriv,tap_jk_deriv,scl,tmp_z,drij,drik,drjk
            real(8) :: r_nl(1:3),feat_val,rcut,cos_angle_val,rcut_ft

            if (set_neigh_info(conf)%threebody(atm)%n.eq.0) then
                !* no 3body interactions for this central atom
                do ft=1,feature_params%num_features
                    if (.not.feature_IsThreeBody(feature_params%info(ft)%ftype)) then
                        cycle
                    end if
                    data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%n = 0
                end do
                return
            end if

            if (calculate_property("forces").and.atom_neigh_info_needs_updating) then
                !* mem allocation
                do ft=1,feature_params%num_features,1
                    if (.not.feature_IsThreeBody(feature_params%info(ft)%ftype)) then
                        cycle
                    end if

                    if (allocated(data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%idx)) then
                        deallocate(data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%idx)
                    end if
                    if (allocated(data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%vec)) then
                        deallocate(data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%vec)
                    end if
                    if (allocated(data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%idx_map)) &
                    &then
                        deallocate(data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%idx_map)
                    end if
                    if (calculate_property("stress")) then
                        if (allocated(data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%&
                        &stress)) then
                            deallocate(data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%stress)
                        end if
                    end if
                end do !* end loop over features

                contrib_atms(1) = atm
                idx_map = -1
                cntr = 1

                do neigh=1,set_neigh_info(conf)%threebody(atm)%n,1
                    do ii=1,2
                        if ( int_in_intarray(set_neigh_info(conf)%threebody(atm)%idx(ii,neigh),&
                        &contrib_atms(1:cntr),arg) ) then
                            !* Local atom already in list, note corresponding idx in contrib_atms
                            idx_map(ii,neigh) = arg
                            cycle
                        else
                            cntr = cntr + 1
                            !* note this local atom contributes to this feature for atom
                            contrib_atms(cntr) = set_neigh_info(conf)%threebody(atm)%idx(ii,neigh)

                            idx_map(ii,neigh) = cntr
                        end if
                    end do !* end loop ii
                end do !* end loop over neighbours
                
                do ft=1,feature_params%num_features,1
                    if (.not.feature_IsThreeBody(feature_params%info(ft)%ftype)) then
                        cycle
                    end if

                    allocate(data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%idx_map(1:2,&
                            &1:set_neigh_info(conf)%threebody(atm)%n))
                    allocate(data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%idx(1:cntr))
                    allocate(data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%vec(1:3,1:cntr))
                    if (calculate_property("stress")) then
                        allocate(data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%&
                                &stress(1:3,1:3,1:cntr))
                    end if
                    
                    !* number of atoms in local cell contributing to feature (including central atom
                    data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%n = cntr

                    ! CAN LAPACK THIS
                    data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%idx(:) = &
                            &contrib_atms(1:cntr)

                    ! CAN LAPACK THIS
                    data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%idx_map = idx_map
                end do !* end loop over features
            end if !* recalculate non-local -> local index map for forces+stress
            
            !* zero all features   
            do ft=1,feature_params%num_features,1
                if (.not.feature_IsThreeBody(feature_params%info(ft)%ftype)) then
                    cycle
                end if
           
                if (calculate_property("forces")) then
                    data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%vec = 0.0d0
                    if (calculate_property("stress")) then
                        data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%stress = 0.0d0
                    end if
                end if
                data_sets(set_type)%configs(conf)%x(ft+1,atm) = 0.0d0
            end do !* end loop over features

            !* max 3body cut off
            rcut = maxrcut(2)

            do bond=1,set_neigh_info(conf)%threebody(atm)%n,1
                drij = set_neigh_info(conf)%threebody(atm)%dr(1,bond)
                drik = set_neigh_info(conf)%threebody(atm)%dr(2,bond)
                drjk = set_neigh_info(conf)%threebody(atm)%dr(3,bond)
                
                if ((drij.gt.rcut).or.(drik.gt.rcut)) then
                    cycle
                end if
                
                cos_angle_val = set_neigh_info(conf)%threebody(atm)%cos_ang(bond)
                
                if (speedup_applies("threebody_rcut")) then
                    tap_ij = set_neigh_info(conf)%threebody(atm)%dr_taper(1,bond)
                    tap_ik = set_neigh_info(conf)%threebody(atm)%dr_taper(2,bond)
                    tap_jk = set_neigh_info(conf)%threebody(atm)%dr_taper(3,bond)
                    tap_ij_deriv = set_neigh_info(conf)%threebody(atm)%dr_taper_deriv(1,bond)
                    tap_ik_deriv = set_neigh_info(conf)%threebody(atm)%dr_taper_deriv(2,bond)
                    tap_jk_deriv = set_neigh_info(conf)%threebody(atm)%dr_taper_deriv(3,bond)
                end if
                
                do ft=1,feature_params%num_features,1
                    ftype = feature_params%info(ft)%ftype
                    
                    if (.not.feature_params%info(ft)%is_threebody) then
                        cycle
                    end if
                    
                    rcut_ft = feature_params%info(ft)%rcut
                    
                    if ((drij.gt.rcut_ft).or.(drik.gt.rcut_ft)) then
                        cycle
                    end if
                    if (.not.feat_doesnt_taper_drjk(ft)) then
                        !* drjk is tapered
                        if (drjk.gt.rcut_ft) then
                            cycle
                        end if
                    end if

                    fs     = feature_params%info(ft)%fs
                    eta    = feature_params%info(ft)%eta
                    xi     = feature_params%info(ft)%xi
                    lambda = feature_params%info(ft)%lambda
                    za     = feature_params%info(ft)%za
                    zb     = feature_params%info(ft)%zb
                    scl    = feature_params%info(ft)%scl_cnst
                    if (ftype.eq.featureID_StringToInt("acsf_normal-b3")) then
                        prec = feature_params%info(ft)%prec
                        mean = feature_params%info(ft)%mean
                    end if


                    if (.not.speedup_applies("threebody_rcut")) then
                        tap_ij = taper_1(drij,rcut,fs)
                        tap_ik = taper_1(drik,rcut,fs)
                        tap_jk = taper_1(drjk,rcut,fs)
                        tap_ij_deriv = taper_deriv_1(drij,rcut,fs)
                        tap_ik_deriv = taper_deriv_1(drik,rcut,fs)
                        tap_jk_deriv = taper_deriv_1(drjk,rcut,fs)
                    end if
                    
                    tmp_z = ( (set_neigh_info(conf)%threebody(atm)%z(1,bond)+1.0d0)*&
                            &(set_neigh_info(conf)%threebody(atm)%z(2,bond)+1.0d0) )**zb *&
                            &(set_neigh_info(conf)%threebody(atm)%z_atom+1.0d0)**za

                    if (ftype.eq.featureID_StringToInt("acsf_behler-g4")) then
                        if (speedup_applies("lookup_tables")) then
                            tmp1 = access_lookup(drij**2+drik**2+drjk**2,map_to_tbl_idx(1,ft))*tmp_z
                            tmp2 = tmp1 * access_lookup(cos_angle_val,map_to_tbl_idx(2,ft))

                            feat_val = tmp2 * (tap_ij*tap_ik*tap_jk)
                        else
                            tmp1 = 2.0d0**(1.0d0-xi)*exp(-eta*(drij**2+drik**2+drjk**2))*tmp_z*scl
                            tmp2 = tmp1 * (1.0d0 + lambda*cos_angle_val)**xi 

                            feat_val = tmp2 * (tap_ij*tap_ik*tap_jk) 
                        end if
                    else if (ftype.eq.featureID_StringToInt("acsf_behler-g5")) then
                        if (speedup_applies("lookup_tables")) then
                            tmp1 = access_lookup(drij**2+drik**2,map_to_tbl_idx(1,ft))*tmp_z
                            tmp2 = tmp1 * access_lookup(cos_angle_val,map_to_tbl_idx(2,ft))

                            feat_val = tmp2 * (tap_ij*tap_ik)
                        else
                            tmp1 = 2.0d0**(1.0d0-xi)*exp(-eta*(drij**2+drik**2)) * tmp_z * scl
                            tmp2 = tmp1 * (1.0d0 + lambda*cos_angle_val)**xi 

                            feat_val = tmp2 * (tap_ij*tap_ik) 
                        end if
                    else if (ftype.eq.featureID_StringToInt("acsf_normal-b3")) then
                        x1(1) = drij
                        x1(2) = drik
                        x1(3) = cos_angle_val
                        x2(1) = drik
                        x2(2) = drij
                        x2(3) = cos_angle_val
        
                        tmp1 = func_normal(x1,mean,prec)
                        tmp2 = func_normal(x2,mean,prec)

                        feat_val = (tmp1 + tmp2)*tmp_z*(tap_ij*tap_ik*tap_jk)*scl 
                    end if

                    !* 0th derivative contribution
                    data_sets(set_type)%configs(conf)%x(ft+1,atm) = & 
                            &data_sets(set_type)%configs(conf)%x(ft+1,atm) + feat_val

                    if (calculate_property("forces")) then
                        if (ftype.eq.featureID_StringToInt("acsf_normal-b3")) then
                            call dsymv('u',3,1.0d0,prec,3,x1-mean,1,0.0d0,tmp_vec1,1)
                            call dsymv('u',3,1.0d0,prec,3,x2-mean,1,0.0d0,tmp_vec2,1)
                        end if

                        do zz=1,3,1
                            if (zz.lt.3) then
                                deriv_idx = data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%&
                                        &idx_map(zz,bond)
                            else
                                deriv_idx = 1
                            end if

                            dcosdrz =  set_neigh_info(conf)%threebody(atm)%dcos_dr(:,zz,bond)

                            if (zz.eq.1) then
                                !* zz=jj
                                drijdrz =  set_neigh_info(conf)%threebody(atm)%drdri(:,1,bond)
                                drikdrz =  set_neigh_info(conf)%threebody(atm)%drdri(:,4,bond)
                                drjkdrz = -set_neigh_info(conf)%threebody(atm)%drdri(:,5,bond)
                            else if (zz.eq.2) then
                                !* zz=kk
                                drijdrz =  set_neigh_info(conf)%threebody(atm)%drdri(:,2,bond)
                                drikdrz =  set_neigh_info(conf)%threebody(atm)%drdri(:,3,bond)
                                drjkdrz =  set_neigh_info(conf)%threebody(atm)%drdri(:,5,bond)
                            else if (zz.eq.3) then
                                !* zz=ii
                                drijdrz = -set_neigh_info(conf)%threebody(atm)%drdri(:,1,bond)
                                drikdrz = -set_neigh_info(conf)%threebody(atm)%drdri(:,3,bond)
                                drjkdrz =  set_neigh_info(conf)%threebody(atm)%drdri(:,6,bond)
                            end if

                            if (ftype.eq.featureID_StringToInt("acsf_behler-g4")) then
                                if (speedup_applies("lookup_tables")) then
                                    tmp3 = access_lookup(cos_angle_val,map_to_tbl_idx(3,ft))
                                else
                                    tmp3 = (1.0d0+lambda*cos_angle_val)**(xi-1.0d0)
                                end if

                                dxdr = tap_ij*tap_ik*tap_jk*lambda*xi*&
                                    &tmp3*dcosdrz*tmp1 +&
                                    &(tap_ik*tap_jk*(tap_ij_deriv - 2.0d0*eta*tap_ij*drij)*drijdrz+&
                                    & tap_ij*tap_jk*(tap_ik_deriv - 2.0d0*eta*tap_ik*drik)*drikdrz+&
                                    & tap_ij*tap_ik*(tap_jk_deriv - 2.0d0*eta*tap_jk*drjk)*drjkdrz)*&
                                    &tmp2
                            else if (ftype.eq.featureID_StringToInt("acsf_behler-g5")) then
                                if (speedup_applies("lookup_tables")) then
                                    tmp3 = access_lookup(cos_angle_val,map_to_tbl_idx(3,ft))
                                else
                                    tmp3 = (1.0d0+lambda*cos_angle_val)**(xi-1.0d0)
                                end if
                                
                                dxdr = tmp1*(tap_ij*tap_ik)*lambda*xi*&
                                    &tmp3*dcosdrz + &
                                    &(tap_ik*(tap_ij_deriv - 2.0d0*eta*tap_ij*drij)*drijdrz +&
                                    &tap_ij*(tap_ik_deriv - 2.0d0*eta*tap_ik*drik)*drikdrz )*tmp2
                            else if (ftype.eq.featureID_StringToInt("acsf_normal-b3")) then
                                norm_tmp1(:,1) = drijdrz
                                norm_tmp1(:,2) = drikdrz
                                norm_tmp1(:,3) = dcosdrz
                                norm_tmp2(:,1) = drikdrz
                                norm_tmp2(:,2) = drijdrz
                                norm_tmp2(:,3) = dcosdrz

                                call dgemv('n',3,3,1.0d0,norm_tmp1,3,tmp_vec1,1,0.0d0,tmp_deriv1,1)
                                call dgemv('n',3,3,1.0d0,norm_tmp2,3,tmp_vec2,1,0.0d0,tmp_deriv2,1)
                               
                                dxdr = scl * ( (tmp1+tmp2)*(tap_ij*tap_jk*tap_ik_deriv*drikdrz + & 
                                &tap_ik*tap_jk*tap_ij_deriv*drijdrz)*tmp_z - &
                                &tap_ij*tap_ik*tap_jk*(tmp1*tmp_deriv1 + &
                                &tmp2*tmp_deriv2)*tmp_z   +  &
                                &tap_ij*tap_ik*tmp_z*(tmp1+tmp2)*tap_jk_deriv*drjkdrz )
                            end if

                            !* force contribution
                            data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%vec(:,deriv_idx)=&
                                &data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%&
                                &vec(:,deriv_idx) + dxdr

                            if (calculate_property("stress")) then
                                !* stress contribution
                                r_nl = set_neigh_info(conf)%threebody(atm)%r_nl(:,zz,bond)

                                call append_stress_contribution(dxdr,r_nl,set_type,conf,&
                                        &atm,ft,deriv_idx)
                            end if
                        end do !* end loop zz over atom differentials
                    end if !* forces
                end do !* end loop over features
            end do !* end loop over bonds to central atom
            
            do ft=1,feature_params%num_features,1
                if (.not.feature_IsThreeBody(feature_params%info(ft)%ftype)) then
                    cycle
                end if
                
                if (updating_net_weights_only.and.calculate_property("forces")) then
                    !* only useful for recomputing loss wrt basis function params
                    if (allocated(data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%idx_map)) then
                        deallocate(data_sets(set_type)%configs(conf)%x_deriv(ft,atm)%idx_map)
                    end if
                end if
            end do
        end subroutine threebody_atom_contribution

        subroutine shift_x(set_type,conf)
            implicit none

            !* args
            integer,intent(in) :: set_type,conf
            
            !* scratch
            integer :: atm,ft
            real(8) :: add_cnst(1:feature_params%num_features)

            do ft=1,feature_params%num_features
                add_cnst(ft) = feature_params%info(ft)%add_cnst
            end do

            do atm=1,data_sets(set_type)%configs(conf)%n,1
                do ft=1,feature_params%num_features
                    data_sets(set_type)%configs(conf)%x(ft+1,atm) = data_sets(set_type)%&
                            &configs(conf)%x(ft+1,atm) + add_cnst(ft)
                end do
            end do
        end subroutine shift_x
        



        subroutine check_performance_criteria()
            use util, only : scalar_equal
            
            implicit none

            integer :: set_type,conf,atm,ft,ftype
            real(8) :: zatom,za,zb
            logical :: multiple_elements_present
            real(8) :: z_consts(1:2)
            logical :: z_params_are_different = .false.

            zatom = -1.0d0
            multiple_elements_present = .false.
            z_consts = -1000.0d0

            !* check for single element case
            do set_type=1,size(data_sets)
                if (.not.allocated(data_sets(set_type)%configs)) then
                    cycle
                end if

                do conf=1,data_sets(set_type)%nconf,1
                    do atm=1,data_sets(set_type)%configs(conf)%n,1
                        if (scalar_equal(zatom,-1.0d0,dble(1e-15),dble(1e-15),.false.)) then
                            zatom = data_sets(set_type)%configs(conf)%z(atm)
                        else if (.not.scalar_equal(zatom,data_sets(set_type)%configs(conf)%z(atm),&
                        &dble(1e-15),dble(1e-15),.false.)) then
                            multiple_elements_present = .true.
                            exit
                        end if
                    end do 
                end do
            end do

            if (.not.multiple_elements_present) then
                call activate_performance_option("single_element")

                !* pre-compute Z weighting for appropriate features
                do ft=1,feature_params%num_features
                    ftype = feature_params%info(ft)%ftype
                    if (ftype.ne.featureID_StringToInt("atomic_number")) then
                        za = feature_params%info(ft)%za
                        zb = feature_params%info(ft)%zb

                        if (feature_IsTwoBody(ftype)) then
                            feature_params%info(ft)%z_single_element = &
                                    &atomic_weighting_twobody_calc(zatom,zatom,za,zb)

                            !* store basis params
                            if (z_consts(1).gt.-1000.0d0) then
                                if (.not.scalar_equal(z_consts(1),za,dble(1e-15),dble(1e-15),&
                                &.false.)) then
                                    z_params_are_different = .true.
                                else if (.not.scalar_equal(z_consts(2),zb,dble(1e-15),dble(1e-15),&
                                &.false.)) then
                                    z_params_are_different = .true.
                                end if
                            else
                                z_consts(1) = za
                                z_consts(2) = zb
                            end if

                        else if (feature_IsThreeBody(ftype)) then
                            feature_params%info(ft)%z_single_element = &
                                    &atomic_weighting_threebody_calc(zatom,zatom,zatom,za,zb)
                            
                            !* store basis params
                            if (z_consts(1).gt.-1000.0d0) then
                                if (.not.scalar_equal(z_consts(1),za,dble(1e-15),dble(1e-15),&
                                &.false.)) then
                                    z_params_are_different = .true.
                                else if (.not.scalar_equal(z_consts(2),zb,dble(1e-15),dble(1e-15),&
                                &.false.)) then
                                    z_params_are_different = .true.
                                end if
                            else
                                z_consts(1) = za
                                z_consts(2) = zb
                            end if
                        else
                            call error("check_performance_criteria","Implementation error")
                        end if
                    end if
                end do

                if (.not.z_params_are_different) then
                    !* all features have same atomic number factor
                    call activate_performance_option("single_element_all_equal")
                end if
            end if

        end subroutine check_performance_criteria
end module

