module features
    use config
    use feature_config
    use feature_util
    use tapering, only : taper_1,taper_deriv_1
   
    implicit none

    external :: dsymv

    contains
        subroutine calculate_features(scale_features,parallel)
            implicit none

            logical,intent(in) :: parallel,scale_features

            integer :: set_type

            do set_type=1,2
                call calculate_features_singleset(set_type,.true.,scale_features,parallel)
            end do
        end subroutine calculate_features
        
        subroutine calculate_features_singleset(set_type,derivatives,scale_features,parallel)
            use omp_lib
            
            implicit none

            integer,intent(in) :: set_type
            logical,intent(in) :: derivatives,scale_features,parallel
            
            real(8),allocatable :: ultra_cart(:,:)
            real(8),allocatable :: ultra_z(:)
            integer,allocatable :: ultra_idx(:)
            real(8) :: mxrcut
            logical :: calc_threebody
            integer :: conf

            !* openMP variables
            integer :: thread_start,thread_end,thread_idx,num_threads
            integer :: dconf

            !* max cut off of all interactions
            mxrcut = maxrcut(0)
            
            !* whether threebody interactions are present
            calc_threebody = threebody_features_present()
            
            !* set whether or not to calculate feature derivatives
            calc_feature_derivatives = derivatives
           
            if (parallel) then
                !$omp parallel num_threads(omp_get_max_threads()),&
                !$omp& default(shared),&
                !$omp& private(conf,thread_start,thread_end,thread_idx,ultra_cart),&
                !$omp& private(ultra_idx,ultra_z)

                !* thread_idx = [0,num_threads-1]    
                thread_idx = omp_get_thread_num()
                
                num_threads = omp_get_max_threads()

                !* number of confs for thread
                dconf = int(floor(float(data_sets(set_type)%nconf)/float(num_threads))) 

                thread_start = thread_idx*dconf + 1

                if (thread_idx.eq.num_threads-1) then
                    thread_end = data_sets(set_type)%nconf 
                else
                    thread_end = (thread_idx+1)*dconf
                end if

                do conf=thread_start,thread_end,1

                    call get_ultracell(mxrcut,5000,set_type,conf,&
                            &ultra_cart,ultra_idx,ultra_z)

                    !* always calc. two-body info for features
                    call calculate_twobody_info(set_type,conf,ultra_cart,ultra_z,ultra_idx)
                
                    if (calc_threebody) then
                        !* calc. threebody info
                        call calculate_threebody_info(set_type,conf,ultra_cart,ultra_z,ultra_idx)
                    end if
                    

                    !* calculate features and their derivatives
                    call calculate_all_features(set_type,conf)

                    deallocate(ultra_z)
                    deallocate(ultra_idx)
                    deallocate(ultra_cart)
                    deallocate(feature_isotropic)
                    if (calc_threebody) then
                        deallocate(feature_threebody_info)
                    end if
                end do !* end loop over configurations

                !$omp end parallel
            else
                do conf=1,data_sets(set_type)%nconf

                    call get_ultracell(mxrcut,5000,set_type,conf,&
                            &ultra_cart,ultra_idx,ultra_z)

                    !* always calc. two-body info for features
                    call calculate_twobody_info(set_type,conf,ultra_cart,ultra_z,ultra_idx)
                
                    if (calc_threebody) then
                        !* calc. threebody info
                        call calculate_threebody_info(set_type,conf,ultra_cart,ultra_z,ultra_idx)
                    end if
                    

                    !* calculate features and their derivatives
                    call calculate_all_features(set_type,conf)

                    deallocate(ultra_z)
                    deallocate(ultra_idx)
                    deallocate(ultra_cart)
                    deallocate(feature_isotropic)
                    if (calc_threebody) then
                        deallocate(feature_threebody_info)
                    end if
                end do !* end loop over configurations
            end if !* end if parallel section

            if (scale_features) then
                call scale_set_features(set_type)
            end if
        end subroutine calculate_features_singleset

        subroutine calculate_distance_distributions(set_type,sample_rate,twobody_dist,threebody_dist,&
                &num_two,num_three)
            implicit none

            !* args
            integer,intent(in) :: set_type
            real(8),intent(in) :: sample_rate(1:2)
            real(8),intent(out) :: twobody_dist(:)
            real(8),intent(out) :: threebody_dist(:,:)
            integer,intent(out) :: num_two,num_three


            !* scratch
            integer :: conf,atm,bond,dim_1,dim_2(1:2)
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
            
            do conf=1,data_sets(set_type)%nconf
                call get_ultracell(mxrcut,5000,set_type,conf,&
                        &ultra_cart,ultra_idx,ultra_z)

                !* always calc. two-body info for features
                call calculate_twobody_info(set_type,conf,ultra_cart,ultra_z,ultra_idx)
            
                do atm=1,data_sets(set_type)%configs(conf)%n
                    if (feature_isotropic(atm)%n.gt.0) then
                        do bond=1,feature_isotropic(atm)%n,1
                            if ((abs(sample_rate(1)-1.0d0).lt.1e-10).or.(rand().lt.sample_rate(1))) then
                                !* book keeping
                                num_two = num_two + 1
                               
                                if (num_two.gt.dim_1) then
                                    call error("calculate_distance_distributions",&
                                            &"two-body buffer too small, increase or decrease sample rate.")
                                end if
                                
                                !* store atom-atom distance from this bond
                                twobody_dist(num_two) = feature_isotropic(atm)%dr(bond)
                            end if
                        end do
                    end if
                end do

                if (calc_threebody) then
                    !* calc. threebody info
                    call calculate_threebody_info(set_type,conf,ultra_cart,ultra_z,ultra_idx)
                
                    do atm=1,data_sets(set_type)%configs(conf)%n
                        if (feature_threebody_info(atm)%n.gt.0) then
                            do bond=1,feature_threebody_info(atm)%n,1
                                if ((abs(sample_rate(2)-1.0d0).lt.1e-10).or.(rand().lt.sample_rate(2))) then
                                    !* book keeping
                                    num_three = num_three + 1
                                
                                    if (num_three.gt.dim_2(2)) then
                                        call error("calculate_distance_distributions",&
                                            &"three-body buffer too small, increase or decrease sample rate.")
                                    end if

                                    threebody_dist(1:2,num_three) = feature_threebody_info(atm)%dr(1:2,bond)
                                    threebody_dist(3,num_three) = feature_threebody_info(atm)%cos_ang(bond)
                                end if
                            end do !* end loop over bonds
                        end if 
                    end do !* loop over local atoms
                end if
                   
                                  
                    
                deallocate(ultra_z)
                deallocate(ultra_idx)
                deallocate(ultra_cart)
                deallocate(feature_isotropic)
                
                if (calc_threebody) then
                    deallocate(feature_threebody_info)
                end if
            end do !* end loop over confs
        end subroutine calculate_distance_distributions

        subroutine calculate_all_features(set_type,conf)
            implicit none

            integer,intent(in) :: set_type,conf
        
            !* scratch
            integer :: atm,ft

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

            !* feature int id
            ftype = feature_params%info(idx)%ftype
            
            if (ftype.eq.0) then
                call feature_atomicnumber_1(set_type,conf,atm,idx)
            else if (feature_IsTwoBody(ftype)) then
                call feature_twobody(set_type,conf,atm,idx)
            else 
                call feature_threebody(set_type,conf,atm,idx)
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
            integer :: idx_to_contrib(1:feature_isotropic(atm)%n)
            logical :: zero_neighbours 
            real(8) :: rcut
            
            zero_neighbours = .true.

            !* weight bias takes 1st element of feature vector
            arr_idx = ft_idx + 1

            !* interaction cut off
            rcut = feature_params%info(ft_idx)%rcut

            !* type of interaction
            ftype = feature_params%info(ft_idx)%ftype

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
            
            do ii=1,feature_isotropic(atm)%n
                if (feature_isotropic(atm)%dr(ii).le.rcut) then
                    !* contributing interaction
                    if (ftype.eq.featureID_StringToInt("acsf_behler-g1")) then
                        call feature_behler_g1(atm,ii,ft_idx,&
                                &data_sets(set_type)%configs(conf)%x(arr_idx,atm))

                        if (calc_feature_derivatives) then
                            call feature_behler_g1_deriv(atm,ii,ft_idx,&
                                &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%&
                                &vec(1:3,idx_to_contrib(ii)))
                        end if
                    else if (ftype.eq.featureID_StringToInt("acsf_behler-g2")) then
                        call feature_behler_g2(atm,ii,ft_idx,&
                                &data_sets(set_type)%configs(conf)%x(arr_idx,atm))

                        if (calc_feature_derivatives) then
                            call feature_behler_g2_deriv(atm,ii,ft_idx,&
                                &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%&
                                &vec(1:3,idx_to_contrib(ii)))
                        end if
                    else if (ftype.eq.featureID_StringToInt("acsf_normal-b2")) then
                        call feature_normal_iso(atm,ii,ft_idx,&
                                &data_sets(set_type)%configs(conf)%x(arr_idx,atm))

                        if (calc_feature_derivatives) then
                            call feature_normal_iso_deriv(atm,ii,ft_idx,&
                                    &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%&
                                    &vec(1:3,idx_to_contrib(ii)))
                        end if
                    else if (ftype.eq.featureID_StringToInt("devel_iso")) then
                        call feature_iso_devel(atm,ii,ft_idx,&
                                &data_sets(set_type)%configs(conf)%x(arr_idx,atm))
                        
                        if (calc_feature_derivatives) then
                            call feature_iso_devel_deriv(atm,ii,ft_idx,&
                                    &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%&
                                    &vec(1:3,idx_to_contrib(ii)))
                        end if
                    end if
                end if
            end do
            
            !* derivative wrt. central atm
            if (calc_feature_derivatives) then 
                if (ftype.eq.featureID_StringToInt("acsf_behler-g1")) then
                    call feature_behler_g1_deriv(atm,0,ft_idx,&
                            &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(1:3,1))
                else if (ftype.eq.featureID_StringToInt("acsf_behler-g2")) then
                    call feature_behler_g2_deriv(atm,0,ft_idx,&
                            &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(1:3,1))
                else if (ftype.eq.featureID_StringToInt("acsf_normal-b2")) then
                    call feature_normal_iso_deriv(atm,0,ft_idx,&
                            &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(1:3,1))
                else if (ftype.eq.featureID_StringToInt("devel_iso")) then
                    call feature_iso_devel_deriv(atm,0,ft_idx,&
                            &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(1:3,1))
                end if
            end if
        end subroutine feature_twobody

        subroutine feature_threebody(set_type,conf,atm,ft_idx)
            implicit none

            integer,intent(in) :: set_type,conf,atm,ft_idx

            real(8) :: rcut
            integer :: ii,jj,arr_idx,cntr,ftype
            integer :: contrib_atms(1:data_sets(set_type)%configs(conf)%n)
            integer :: idx_to_contrib(1:2,1:feature_threebody_info(atm)%n)
            integer :: arg
            logical :: not_null
            logical,allocatable :: bond_contributes(:)

            rcut = feature_params%info(ft_idx)%rcut

            !* weight bias included by null feature coordinate
            arr_idx = ft_idx + 1

            not_null = .false.
            
            !* feature type
            ftype = feature_params%info(ft_idx)%ftype

            !* is three-body term within rcut?
            allocate(bond_contributes(feature_threebody_info(atm)%n))

            !* check given (feature,atom) has three-body terms within rcut
            do ii=1,feature_threebody_info(atm)%n,1
                if (maxval(feature_threebody_info(atm)%dr(:,ii)).le.rcut) then
                    bond_contributes(ii) = .true.
                else
                    bond_contributes(ii) = .false.
                end if
            end do

            if ( (any(bond_contributes).neqv..true.).or.(feature_threebody_info(atm)%n.eq.0) ) then
                !* zero neighbours within rcut
                data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%n=0
                data_sets(set_type)%configs(conf)%x(arr_idx,atm) = 0.0d0
                return
            end if
            
            
            !* idx of central atom
            contrib_atms(1) = atm  


            !* NULL value for terms not within rcut
            idx_to_contrib(:,:) = -1

            cntr = 1
            do ii=1,feature_threebody_info(atm)%n,1
                if (bond_contributes(ii).neqv..true.) then
                    cycle
                end if

                do jj=1,2
                    if ( int_in_intarray(feature_threebody_info(atm)%idx(jj,ii),contrib_atms(1:cntr),arg) ) then
                        !* Local atom already in list, note corresponding idx in contrib_atms
                        idx_to_contrib(jj,ii) = arg
                        cycle
                    else 
                        cntr = cntr + 1
                        !* note this local atom contributes to this feature for atom
                        contrib_atms(cntr) = feature_threebody_info(atm)%idx(jj,ii)
                        
                        idx_to_contrib(jj,ii) = cntr
                    end if
                end do !* end loop over 2 neighbouring atoms                    
            end do !* end loop over threebody terms
            
            allocate(data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%idx(cntr))
            allocate(data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(3,cntr))
            
            !* number of atoms in local cell contributing to feature (including central atom)
            data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%n = cntr
            
            !* local indices of atoms contributing to feature
            data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%idx(:) = contrib_atms(1:cntr)

            !* zero features
            data_sets(set_type)%configs(conf)%x(arr_idx,atm) = 0.0d0
            data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(:,:) = 0.0d0

            do ii=1,feature_threebody_info(atm)%n,1
                if(bond_contributes(ii).neqv..true.) then
                    cycle
                end if

                if (ftype.eq.featureID_StringToInt("acsf_behler-g4")) then
                    call feature_behler_g4(set_type,conf,atm,ft_idx,ii)

                    if (calc_feature_derivatives) then
                        call feature_behler_g4_deriv(set_type,conf,atm,ft_idx,ii,idx_to_contrib(:,ii)) 
                    end if
                else if (ftype.eq.featureID_StringToInt("acsf_behler-g5")) then
                    call feature_behler_g5(set_type,conf,atm,ft_idx,ii)

                    if (calc_feature_derivatives) then
                        call feature_behler_g5_deriv(set_type,conf,atm,ft_idx,ii,idx_to_contrib(:,ii)) 
                    end if
                else if (ftype.eq.featureID_StringToInt("acsf_normal-b3")) then
                    call feature_normal_threebody(set_type,conf,atm,ft_idx,ii)

                    if (calc_feature_derivatives) then
                        call feature_normal_threebody_deriv(set_type,conf,atm,ft_idx,ii,&
                                &idx_to_contrib(:,ii)) 
                    end if
                end if
            end do !* end loop ii over three body terms
            
            deallocate(bond_contributes)
        
        end subroutine feature_threebody
        
        subroutine feature_behler_g1(atm,neigh_idx,ft_idx,current_val)
            implicit none

            integer,intent(in) :: atm,neigh_idx,ft_idx
            real(8),intent(inout) :: current_val

            !* scratch
            real(8) :: dr,tmp2,tmp3,za,zb,rcut,fs
           
            !* atom-neigh_idx distance 
            dr  = feature_isotropic(atm)%dr(neigh_idx)
            
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
            tmp2 = taper_1(dr,rcut,fs)
        
            !* atomic numbers
            tmp3 = (feature_isotropic(atm)%z_atom+1.0d0)**za * &
                    &(feature_isotropic(atm)%z(neigh_idx)+1.0d0)**zb

            
            current_val = current_val + tmp2*tmp3
        end subroutine feature_behler_g1
        
        subroutine feature_behler_g1_deriv(atm,neigh_idx,ft_idx,deriv_vec)
            implicit none

            integer,intent(in) :: atm,neigh_idx,ft_idx
            real(8),intent(inout) :: deriv_vec(1:3)

            !* scratch
            real(8) :: dr_scl,dr_vec(1:3),tap_deriv,tap,tmp1,tmp2
            real(8) :: fs,rcut,tmpz
            real(8) :: za,zb
            integer :: ii,lim1,lim2

            !* symmetry function params
            za   = feature_params%info(ft_idx)%za
            zb   = feature_params%info(ft_idx)%zb
            fs   = feature_params%info(ft_idx)%fs
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
                if (atm.eq.feature_isotropic(atm)%idx(ii)) then
                    ! dr_vec =  d (r_i + const - r_i ) / d r_i = 0
                    cycle
                end if
                
                !* atom-atom distance
                dr_scl = feature_isotropic(atm)%dr(ii)
                
                if (dr_scl.gt.rcut) then
                    cycle
                end if

                !* (r_neighbour - r_centralatom)/dr_scl
                dr_vec(:) = feature_isotropic(atm)%drdri(:,ii)
                
                !* tapering
                tap = taper_1(dr_scl,rcut,fs)
                tap_deriv = taper_deriv_1(dr_scl,rcut,fs)

                !* atomic numbers
                tmpz = (feature_isotropic(atm)%z_atom+1.0d0)**za * (feature_isotropic(atm)%z(ii)+1.0d0)**zb

                tmp1 = tap_deriv
                
                deriv_vec(:) = deriv_vec(:) + dr_vec(:)*tmp1*tmp2*tmpz
            end do
        end subroutine feature_behler_g1_deriv
        
        subroutine feature_behler_g2(atm,neigh_idx,ft_idx,current_val)
            implicit none

            integer,intent(in) :: atm,neigh_idx,ft_idx
            real(8),intent(inout) :: current_val

            !* scratch
            real(8) :: dr,tmp1,tmp2,tmp3,za,zb,rcut,eta,rs,fs
           
            !* atom-neigh_idx distance 
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
            tmp3 = (feature_isotropic(atm)%z_atom+1.0d0)**za * &
                    &(feature_isotropic(atm)%z(neigh_idx)+1.0d0)**zb

            current_val = current_val + tmp1*tmp2*tmp3
        end subroutine feature_behler_g2
      
        subroutine feature_behler_g2_deriv(atm,neigh_idx,ft_idx,deriv_vec)
            implicit none

            integer,intent(in) :: atm,neigh_idx,ft_idx
            real(8),intent(inout) :: deriv_vec(1:3)

            !* scratch
            real(8) :: dr_scl,dr_vec(1:3),tap_deriv,tap,tmp1,tmp2
            real(8) :: rs,fs,rcut,tmpz
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
                if (atm.eq.feature_isotropic(atm)%idx(ii)) then
                    ! dr_vec =  d (r_i + const - r_i ) / d r_i = 0
                    cycle
                end if
                
                !* atom-atom distance
                dr_scl = feature_isotropic(atm)%dr(ii)

                !* (r_neighbour - r_centralatom)/dr_scl
                dr_vec(:) = feature_isotropic(atm)%drdri(:,ii)
                
                !* tapering
                tap = taper_1(dr_scl,rcut,fs)
                tap_deriv = taper_deriv_1(dr_scl,rcut,fs)

                !* atomic numbers
                tmpz = (feature_isotropic(atm)%z_atom+1.0d0)**za * (feature_isotropic(atm)%z(ii)+1.0d0)**zb

                tmp1 =  exp(-eta*(dr_scl-rs)**2)  *  (tap_deriv - &
                        &2.0d0*eta*(dr_scl-rs)*tap) 
                
                deriv_vec(:) = deriv_vec(:) + dr_vec(:)*tmp1*tmp2*tmpz
            end do
        end subroutine feature_behler_g2_deriv
       
        subroutine feature_behler_g4(set_type,conf,atm,ft_idx,bond_idx)
            implicit none

            !* args
            integer,intent(in) :: atm,ft_idx,bond_idx,set_type,conf

            !* scratch
            real(8) :: xi,eta,lambda,fs,rcut,za,zb
            real(8) :: tmp_atmz,tmp_taper
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
            drij = feature_threebody_info(atm)%dr(1,bond_idx)
            drik = feature_threebody_info(atm)%dr(2,bond_idx)
            drjk = feature_threebody_info(atm)%dr(3,bond_idx)

            if ( (drij.gt.rcut).or.(drik.gt.rcut).or.(drjk.gt.rcut) ) then
                return
            end if

            cos_angle = feature_threebody_info(atm)%cos_ang(bond_idx)

            !* atomic number term
            tmp_atmz = (feature_threebody_info(atm)%z_atom+1.0d0)**za *&
                    &( (feature_threebody_info(atm)%z(1,bond_idx)+1.0d0)*&
                    &(feature_threebody_info(atm)%z(2,bond_idx)+1.0d0) )**zb

            !* taper term
            tmp_taper = taper_1(drij,rcut,fs)*taper_1(drik,rcut,fs)*taper_1(drjk,rcut,fs)

            data_sets(set_type)%configs(conf)%x(ft_idx+1,atm) = data_sets(set_type)%configs(conf)%x(ft_idx+1,atm)&
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
            real(8) :: drij,drik,drjk,cos_angle,tmp_z
            integer :: zz,deriv_idx
            real(8) :: tmp_feature1,tmp_feature2,tap_ij,tap_jk,tap_ik
            real(8) :: tap_ij_deriv,tap_ik_deriv,tap_jk_deriv
            real(8) :: dcosdrz(1:3),drijdrz(1:3),drikdrz(1:3),drjkdrz(1:3)

            !* feature parameters
            rcut   = feature_params%info(ft_idx)%rcut
            eta    = feature_params%info(ft_idx)%eta
            xi     = feature_params%info(ft_idx)%xi
            lambda = feature_params%info(ft_idx)%lambda
            fs     = feature_params%info(ft_idx)%fs
            za     = feature_params%info(ft_idx)%za
            zb     = feature_params%info(ft_idx)%zb

            !* atom-atom distances
            drij = feature_threebody_info(atm)%dr(1,bond_idx)
            drik = feature_threebody_info(atm)%dr(2,bond_idx)
            drjk = feature_threebody_info(atm)%dr(3,bond_idx)
            
            if ( (drij.gt.rcut).or.(drik.gt.rcut).or.(drjk.gt.rcut) ) then
                return
            end if

            cos_angle = feature_threebody_info(atm)%cos_ang(bond_idx)

            !* tapering
            tap_ij = taper_1(drij,rcut,fs)
            tap_ik = taper_1(drik,rcut,fs)
            tap_jk = taper_1(drjk,rcut,fs)
            tap_ij_deriv = taper_deriv_1(drij,rcut,fs)
            tap_ik_deriv = taper_deriv_1(drik,rcut,fs)
            tap_jk_deriv = taper_deriv_1(drjk,rcut,fs)

            !* atomic numbers
            tmp_z = ( (feature_threebody_info(atm)%z(1,bond_idx)+1.0d0)*&
                    &(feature_threebody_info(atm)%z(2,bond_idx)+1.0d0) )**zb *&
                    &(feature_threebody_info(atm)%z_atom+1.0d0)**za

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
                dcosdrz =  feature_threebody_info(atm)%dcos_dr(:,zz,bond_idx)
                
                if (zz.eq.1) then
                    ! zz=jj
                    drijdrz =  feature_threebody_info(atm)%drdri(:,1,bond_idx)
                    drikdrz =  feature_threebody_info(atm)%drdri(:,4,bond_idx)
                    drjkdrz = -feature_threebody_info(atm)%drdri(:,5,bond_idx)
                else if (zz.eq.2) then
                    ! zz=kk
                    drijdrz =  feature_threebody_info(atm)%drdri(:,2,bond_idx)
                    drikdrz =  feature_threebody_info(atm)%drdri(:,3,bond_idx)
                    drjkdrz =  feature_threebody_info(atm)%drdri(:,5,bond_idx)
                else if (zz.eq.3) then
                    ! zz=ii
                    drijdrz = -feature_threebody_info(atm)%drdri(:,1,bond_idx)
                    drikdrz = -feature_threebody_info(atm)%drdri(:,3,bond_idx)
                    drjkdrz =  feature_threebody_info(atm)%drdri(:,6,bond_idx)
                end if

                data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(:,deriv_idx) = &
                    &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(:,deriv_idx) + & 
                    &tap_ij*tap_ik*tap_jk*lambda*xi*((1.0d0+lambda*cos_angle)**(xi-1.0d0))*&
                    &dcosdrz*tmp_feature1 +&
                    &(tap_ik*tap_jk*(tap_ij_deriv - 2.0d0*eta*tap_ij*drij)*drijdrz +&
                    &tap_ij*tap_jk*(tap_ik_deriv - 2.0d0*eta*tap_ik*drik)*drikdrz +&
                    &tap_ij*tap_ik*(tap_jk_deriv - 2.0d0*eta*tap_jk*drjk)*drjkdrz  )*tmp_feature2
                
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

            !* feature parameters
            rcut   = feature_params%info(ft_idx)%rcut
            eta    = feature_params%info(ft_idx)%eta
            xi     = feature_params%info(ft_idx)%xi
            lambda = feature_params%info(ft_idx)%lambda
            fs     = feature_params%info(ft_idx)%fs
            za     = feature_params%info(ft_idx)%za
            zb     = feature_params%info(ft_idx)%zb

            !* atom-atom distances
            drij = feature_threebody_info(atm)%dr(1,bond_idx)
            drik = feature_threebody_info(atm)%dr(2,bond_idx)
            
            if ( (drij.gt.rcut).or.(drik.gt.rcut) ) then
                return
            end if

            cos_angle = feature_threebody_info(atm)%cos_ang(bond_idx)

            !* atomic number term
            tmp_atmz = (feature_threebody_info(atm)%z_atom+1.0d0)**za *&
                    &( (feature_threebody_info(atm)%z(1,bond_idx)+1.0d0)*&
                    &(feature_threebody_info(atm)%z(2,bond_idx)+1.0d0) )**zb

            !* taper term
            tmp_taper = taper_1(drij,rcut,fs)*taper_1(drik,rcut,fs)

            data_sets(set_type)%configs(conf)%x(ft_idx+1,atm) = data_sets(set_type)%configs(conf)%x(ft_idx+1,atm)&
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
            real(8) :: tap_ij_deriv,tap_ik_deriv
            real(8) :: dcosdrz(1:3),drijdrz(1:3),drikdrz(1:3)

            !* feature parameters
            rcut   = feature_params%info(ft_idx)%rcut
            eta    = feature_params%info(ft_idx)%eta
            xi     = feature_params%info(ft_idx)%xi
            lambda = feature_params%info(ft_idx)%lambda
            fs     = feature_params%info(ft_idx)%fs
            za     = feature_params%info(ft_idx)%za
            zb     = feature_params%info(ft_idx)%zb

            !* atom-atom distances
            drij = feature_threebody_info(atm)%dr(1,bond_idx)
            drik = feature_threebody_info(atm)%dr(2,bond_idx)
            
            if ( (drij.gt.rcut).or.(drik.gt.rcut) ) then
                return
            end if

            cos_angle = feature_threebody_info(atm)%cos_ang(bond_idx)

            !* tapering
            tap_ij = taper_1(drij,rcut,fs)
            tap_ik = taper_1(drik,rcut,fs)
            tap_ij_deriv = taper_deriv_1(drij,rcut,fs)
            tap_ik_deriv = taper_deriv_1(drik,rcut,fs)

            !* atomic numbers
            tmp_z = ( (feature_threebody_info(atm)%z(1,bond_idx)+1.0d0)*&
                    &(feature_threebody_info(atm)%z(2,bond_idx)+1.0d0) )**zb *&
                    &(feature_threebody_info(atm)%z_atom+1.0d0)**za

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
                dcosdrz =  feature_threebody_info(atm)%dcos_dr(:,zz,bond_idx)
                
                if (zz.eq.1) then
                    ! zz=jj
                    drijdrz =  feature_threebody_info(atm)%drdri(:,1,bond_idx)
                    drikdrz =  feature_threebody_info(atm)%drdri(:,4,bond_idx)
                else if (zz.eq.2) then
                    ! zz=kk
                    drijdrz =  feature_threebody_info(atm)%drdri(:,2,bond_idx)
                    drikdrz =  feature_threebody_info(atm)%drdri(:,3,bond_idx)
                else if (zz.eq.3) then
                    ! zz=ii
                    drijdrz = -feature_threebody_info(atm)%drdri(:,1,bond_idx)
                    drikdrz = -feature_threebody_info(atm)%drdri(:,3,bond_idx)
                end if

                data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(:,deriv_idx) = &
                &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(:,deriv_idx) + & 
                    &tmp_feature1*tap_ij*tap_ik*lambda*xi*((1.0d0+lambda*cos_angle)**(xi-1.0d0))*&
                    &dcosdrz +&
                    &(tap_ik*(tap_ij_deriv - 2.0d0*eta*tap_ij*drij)*drijdrz +&
                    &tap_ij*(tap_ik_deriv - 2.0d0*eta*tap_ik*drik)*drikdrz )*tmp_feature2
                
            end do
            
        end subroutine feature_behler_g5_deriv
        
        subroutine feature_normal_iso(atm,neigh_idx,ft_idx,current_val)
            implicit none

            integer,intent(in) :: atm,neigh_idx,ft_idx
            real(8),intent(inout) :: current_val

            !* scratch
            real(8) :: dr,tmp1,tmp2,tmp3,za,zb,rcut,fs,prec
            real(8) :: invsqrt2pi,mean,sqrt_det

            invsqrt2pi = 0.3989422804014327d0

            !* atom-neigh_idx distance 
            dr  = feature_isotropic(atm)%dr(neigh_idx)
           
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
            tmp1 = sqrt_det*invsqrt2pi*exp(-0.5d0*prec*(dr-mean)**2)

            !* tapering
            tmp2 = taper_1(dr,rcut,fs)
        
            !* atomic numbers
            tmp3 = (feature_isotropic(atm)%z_atom+1.0d0)**za * &
                    &(feature_isotropic(atm)%z(neigh_idx)+1.0d0)**zb

            current_val = current_val + tmp1*tmp2*tmp3
        end subroutine feature_normal_iso
        
        subroutine feature_normal_iso_deriv(atm,neigh_idx,ft_idx,deriv_vec)
            implicit none

            integer,intent(in) :: atm,neigh_idx,ft_idx
            real(8),intent(inout) :: deriv_vec(1:3)

            !* scratch
            real(8) :: dr_scl,dr_vec(1:3),tap_deriv,tap,tmp1,tmp2
            real(8) :: fs,rcut,tmpz,prec,mean,sqrt_det
            real(8) :: za,zb,invsqrt2pi,prec_const
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
                lim2 = feature_isotropic(atm)%n
                tmp2 = -1.0d0       !* sign for drij/d r_central
            else
                lim1 = neigh_idx
                lim2 = neigh_idx    
                tmp2 = 1.0d0        !* sign for drij/d r_neighbour
            end if


            !* derivative wrt. central atom itself
            do ii=lim1,lim2,1
                if (atm.eq.feature_isotropic(atm)%idx(ii)) then
                    ! dr_vec =  d (r_i + const - r_i ) / d r_i = 0
                    cycle
                end if
                
                !* atom-atom distance
                dr_scl = feature_isotropic(atm)%dr(ii)

                if (dr_scl.gt.rcut) then
                    cycle
                end if

                !* (r_neighbour - r_centralatom)/dr_scl
                dr_vec(:) = feature_isotropic(atm)%drdri(:,ii)
                
                !* tapering
                tap = taper_1(dr_scl,rcut,fs)
                tap_deriv = taper_deriv_1(dr_scl,rcut,fs)

                !* atomic numbers
                tmpz = (feature_isotropic(atm)%z_atom+1.0d0)**za * (feature_isotropic(atm)%z(ii)+1.0d0)**zb

                tmp1 =  prec_const*exp(-0.5d0*prec*(dr_scl-mean)**2)  *  (tap_deriv - &
                        &prec*(dr_scl-mean)*tap) 
                
                deriv_vec(:) = deriv_vec(:) + dr_vec(:)*tmp1*tmp2*tmpz
            end do
        end subroutine feature_normal_iso_deriv
        
        real(8) function func_normal(x,mean,prec,sqrt_det)
            implicit none
            
            !* args
            real(8),intent(in) :: sqrt_det,x(:),mean(:),prec(:,:)

            !* scratch
            integer :: n
            real(8) :: lwork(1:3),pi_const

            !* sqrt(1/(2 pi))
            pi_const = 0.3989422804014327

            n = size(x)

            !* lwork = prec * (x - mean) (prec is symmetric)
            call dsymv('u',n,1.0d0,prec,n,x-mean,1,0.0d0,lwork,1)

            !* sum_i lwork_i*(x-mean)_i
            func_normal = exp(-0.5d0*ddot(n,x-mean,1,lwork,1)) * (sqrt_det * pi_const)**n
        end function func_normal

        subroutine feature_iso_devel(atm,neigh_idx,ft_idx,current_val)
            use propagate, only : logistic

            implicit none

            integer,intent(in) :: atm,neigh_idx,ft_idx
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
            dr  = feature_isotropic(atm)%dr(neigh_idx)
            tmp_taper = taper_1(dr,r_taper,fs)

            xtilde = const*(dr-mean)
            
            current_val = current_val + logistic(xtilde)*tmp_taper
        end subroutine feature_iso_devel
        
        subroutine feature_iso_devel_deriv(atm,neigh_idx,ft_idx,deriv_vec)
            use propagate, only : logistic,logistic_deriv
            
            implicit none

            integer,intent(in) :: atm,neigh_idx,ft_idx
            real(8),intent(inout) :: deriv_vec(1:3)

            integer :: lim1,lim2,ii
            real(8) :: tmp2,dr_scl,dr_vec(1:3)
            real(8) :: rcuts(1:2),mean,const,r_taper,std
            real(8) :: xtilde,sig,sig_prime,fs
            real(8) :: tap,tap_deriv

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
                lim2 = feature_isotropic(atm)%n
                tmp2 = -1.0d0       !* sign for drij/d r_central
            else
                lim1 = neigh_idx
                lim2 = neigh_idx    
                tmp2 = 1.0d0        !* sign for drij/d r_neighbour
            end if


            !* derivative wrt. central atom itself
            do ii=lim1,lim2,1
                if (atm.eq.feature_isotropic(atm)%idx(ii)) then
                    ! dr_vec =  d (r_i + const - r_i ) / d r_i = 0
                    cycle
                end if
                
                !* atom-atom distance
                dr_scl = feature_isotropic(atm)%dr(ii)

                if (dr_scl.gt.r_taper) then
                    cycle
                end if

                !* (r_neighbour - r_centralatom)/dr_scl
                dr_vec(:) = feature_isotropic(atm)%drdri(:,ii)
                
                !* tapering
                tap = taper_1(dr_scl,r_taper,fs)
                tap_deriv = taper_deriv_1(dr_scl,r_taper,fs)

                sig = logistic(xtilde)
                sig_prime = logistic_deriv(xtilde)*const


                deriv_vec(:) = deriv_vec(:) + (sig*tap_deriv + sig_prime*tap)*dr_vec(:)*tmp2
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
            real(8) :: drij,drik,cos_angle,sqrt_det

            !* feature parameters
            rcut     = feature_params%info(ft_idx)%rcut
            prec    = feature_params%info(ft_idx)%prec
            mean    = feature_params%info(ft_idx)%mean
            fs       = feature_params%info(ft_idx)%fs
            za       = feature_params%info(ft_idx)%za
            zb       = feature_params%info(ft_idx)%zb
            sqrt_det = feature_params%info(ft_idx)%sqrt_det

            !* atom-atom distances
            drij = feature_threebody_info(atm)%dr(1,bond_idx)
            drik = feature_threebody_info(atm)%dr(2,bond_idx)
            
            if ( (drij.gt.rcut).or.(drik.gt.rcut) ) then
                return
            end if

            cos_angle = feature_threebody_info(atm)%cos_ang(bond_idx)

            !* must permute atom order to retain invariance
            x1(1) = drij
            x1(2) = drik
            x1(3) = cos_angle
            x2(1) = drik
            x2(2) = drij
            x2(3) = cos_angle

            !* atomic number term
            tmp_atmz = (feature_threebody_info(atm)%z_atom+1.0d0)**za *&
                    &( (feature_threebody_info(atm)%z(1,bond_idx)+1.0d0)*&
                    &(feature_threebody_info(atm)%z(2,bond_idx)+1.0d0) )**zb

            !* taper term
            tmp_taper = taper_1(drij,rcut,fs)*taper_1(drik,rcut,fs)

            data_sets(set_type)%configs(conf)%x(ft_idx+1,atm) = &
                    &data_sets(set_type)%configs(conf)%x(ft_idx+1,atm)&
                    + (func_normal(x1,mean,prec,sqrt_det) + func_normal(x2,mean,prec,sqrt_det)) * &
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
            real(8) :: x(1:3,1:2)
            real(8) :: drij,drik,cos_angle
            real(8) :: tmp_feat(1:2),tmp_vec(1:3,1:2)
            real(8) :: tmp_deriv1(1:3),tmp_deriv2(1:3)
            real(8) :: dxdr1(1:3,1:3),dxdr2(1:3,1:3)
            real(8) :: dcosdrz(1:3),drijdrz(1:3),drikdrz(1:3),tap_ij,tap_ik
            real(8) :: tap_ij_deriv,tap_ik_deriv,tmpz
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
            drij = feature_threebody_info(atm)%dr(1,bond_idx)
            drik = feature_threebody_info(atm)%dr(2,bond_idx)
            
            if ( (drij.gt.rcut).or.(drik.gt.rcut) ) then
                return
            end if

            cos_angle = feature_threebody_info(atm)%cos_ang(bond_idx)

            !* must permuate atom ordering to retain invariance
            x(1,1) = drij
            x(2,1) = drik
            x(1,2) = drik
            x(2,2) = drij 
            x(3,:) = cos_angle

            !* tapering
            tap_ij = taper_1(drij,rcut,fs)
            tap_ik = taper_1(drik,rcut,fs)
            tap_ij_deriv = taper_deriv_1(drij,rcut,fs)
            tap_ik_deriv = taper_deriv_1(drik,rcut,fs)

            !* atomic numbers
            tmpz = ( (feature_threebody_info(atm)%z(1,bond_idx)+1.0d0)*&
                    &(feature_threebody_info(atm)%z(2,bond_idx)+1.0d0) )**zb *&
                    &(feature_threebody_info(atm)%z_atom+1.0d0)**za

            tmp_feat(1) = func_normal(x(:,1),mean,prec,sqrt_det)
            tmp_feat(2) = func_normal(x(:,2),mean,prec,sqrt_det)

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
                dcosdrz =  feature_threebody_info(atm)%dcos_dr(:,zz,bond_idx)
                
                if (zz.eq.1) then
                    ! zz=jj
                    drijdrz =  feature_threebody_info(atm)%drdri(:,1,bond_idx)
                    drikdrz =  feature_threebody_info(atm)%drdri(:,4,bond_idx)
                else if (zz.eq.2) then
                    ! zz=kk
                    drijdrz =  feature_threebody_info(atm)%drdri(:,2,bond_idx)
                    drikdrz =  feature_threebody_info(atm)%drdri(:,3,bond_idx)
                else if (zz.eq.3) then
                    ! zz=ii
                    drijdrz = -feature_threebody_info(atm)%drdri(:,1,bond_idx)
                    drikdrz = -feature_threebody_info(atm)%drdri(:,3,bond_idx)
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

                data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(:,deriv_idx) = &
                        &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(:,deriv_idx) + & 
                        &sum(tmp_feat)*(tap_ij*tap_ik_deriv*drikdrz + tap_ik*tap_ij_deriv*drijdrz)*tmpz - &
                        &tap_ij*tap_ik*(tmp_feat(1)*tmp_deriv1+tmp_feat(2)*tmp_deriv2)*tmpz
                 
            end do
            
        end subroutine feature_normal_threebody_deriv

end module
