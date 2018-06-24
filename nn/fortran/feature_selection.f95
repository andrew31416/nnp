module feature_selection
    use feature_util, only : get_ultracell, maxrcut, threebody_features_present
    use feature_util, only : calculate_threebody_info,calculate_twobody_info
    use feature_util, only : scale_conf_features
    use tapering, only : taper_1,taper_deriv_1
    use propagate, only : forward_propagate,backward_propagate,calculate_forces
    use propagate, only : calculate_d2ydxdx,d2ydxdx
    use features, only : calculate_all_features
    use io, only : error
    use init, only : allocate_units
    use util, only : parse_array_to_structure,copy_weights_to_nobiasT
    use util, only : allocate_dydx,load_balance_alg_1,scalar_equal
    use feature_config
    use config

    implicit none

    real(8),external :: ddot

    contains
        subroutine loss_feature_jacobian(flat_weights,set_type,scale_features,parallel,jacobian)
            use omp_lib
            
            implicit none

            real(8),intent(in) :: flat_weights(:)
            integer,intent(in) :: set_type
            logical,intent(in) :: scale_features,parallel
            real(8),intent(out) :: jacobian(:)

            !* scratch
            integer :: conf
            type(feature_info) :: gbl_derivs,lcl_derivs
            logical :: original_calc_status
            
            !* openMP variables
            integer :: thread_idx,num_threads,bounds(1:2)
            
            if (num_optimizable_params().ne.size(jacobian)) then
                call error("loss_feature_jacobian",&
                        &"Mismatch between length of Py and F95 jacobian")
            end if 

            original_calc_status = calculate_property("forces")
            if (scalar_equal(loss_const_forces,0.0d0,dble(1e-15),dble(1e-10)**2,.false.)) then
                !* check if forces are included
                call switch_property("forces","off")
            else
                call switch_property("forces","on")
            end if
            
            
            !* need to read in net weights
            call parse_array_to_structure(flat_weights,net_weights)
            call copy_weights_to_nobiasT()

            !* allocate memory for feat. derivs wrt. basis fun. params
            call init_feature_array(gbl_derivs)


            if (.not.allocated(set_neigh_info)) then
                allocate(set_neigh_info(data_sets(set_type)%nconf))
            end if

            if (parallel) then
                !$omp parallel num_threads(omp_get_max_threads()),&
                !$omp& default(shared),&
                !$omp& private(conf,thread_idx,num_threads,bounds),&
                !$omp& private(lcl_derivs)
            
                !* allocatable init
                call init_feature_array(lcl_derivs)

                !* [0,num_threads-1]
                thread_idx = omp_get_thread_num()

                !* number of threads
                num_threads = omp_get_max_threads()

                !* split as evenly as possible
                call load_balance_alg_1(thread_idx,num_threads,data_sets(set_type)%nconf,bounds)

                do conf=bounds(1),bounds(2),1
                    call single_conf_feat_jac(set_type,conf,scale_features,lcl_derivs)
                end do
                
                !$omp critical
                    !* add local (thread) to global contribution
                    call add_feat_param_derivs(gbl_derivs,lcl_derivs,1.0d0)
                !$omp end critical

                !$omp end parallel
            else

                do conf=1,data_sets(set_type)%nconf,1
                    call single_conf_feat_jac(set_type,conf,scale_features,gbl_derivs)
                end do !* end loop over confs
            end if

            if (original_calc_status) then
                call switch_property("forces","on")
            else
                call switch_property("forces","off")
            end if
            
            !* performance flag
            atom_neigh_info_needs_updating = .false.
            
            call parse_feature_format_to_array_jac(gbl_derivs,jacobian)
        end subroutine loss_feature_jacobian

        subroutine single_conf_feat_jac(set_type,conf,scale_features,lcl_feat_derivs)
            implicit none

            integer,intent(in) :: set_type,conf
            type(feature_info),intent(inout) :: lcl_feat_derivs
            logical,intent(in) :: scale_features

            !* scratch
            real(8) :: mxrcut,dr,zatm,zngh,tmpE,invN
            real(8) :: force_norm_const(1:3,1:data_sets(set_type)%configs(conf)%n)
            real(8),allocatable :: ultra_cart(:,:),ultra_z(:)
            integer,allocatable :: ultra_idx(:)
            integer :: atm,neigh,ftype,bond,ft,xx
            logical :: calc_threebody
            type(feature_info) :: force_contribution 
            type(feature_info) :: dxdparam(1:data_sets(set_type)%configs(conf)%n)
            type(feature_info) :: d2xdrdparam(1:data_sets(set_type)%configs(conf)%n,&
                                             &1:data_sets(set_type)%configs(conf)%n,1:3)


            if (.not.allocated(set_neigh_info)) then
                call error("single_conf_feat_jac","set_neigh_info should be initialised")
            end if
            
            !* init param allocatables 
            do atm=1,data_sets(set_type)%configs(conf)%n,1
                call init_feature_array(dxdparam(atm))
                call zero_feature_info(dxdparam(atm))
            end do

            if (calculate_property("forces")) then
                call init_feature_array(force_contribution)
                call zero_feature_info(force_contribution)

                do atm=1,data_sets(set_type)%configs(conf)%n
                    do neigh=1,data_sets(set_type)%configs(conf)%n
                        do xx=1,3
                            call init_feature_array(d2xdrdparam(atm,neigh,xx))
                            call zero_feature_info(d2xdrdparam(atm,neigh,xx))
                        end do
                    end do
                end do
            end if
            
            !* max cut off of all interactions
            mxrcut = maxrcut(0)

            !* whether three body interactions are present
            calc_threebody = threebody_features_present()
           
            if (.not.allocated(set_neigh_info(conf)%twobody)) then
                !* get all nearest neighbours
                call get_ultracell(mxrcut,5000,set_type,conf,ultra_cart,ultra_idx,ultra_z)
                
                !* get atom-neighbour distances
                call calculate_twobody_info(set_type,conf,ultra_cart,ultra_z,ultra_idx)

                if (calc_threebody) then
                    call calculate_threebody_info(set_type,conf,ultra_cart,ultra_z,ultra_idx)
                end if
            end if

            !* compute new features and derivatives if calc_feature_derivatives
            call calculate_all_features(set_type,conf,.true.)
            if (scale_features) then
                !* don't touch feature derivatives wrt atoms if not using forces
                call scale_conf_features(set_type,conf)
            end if

            !* allocate mem. for dydx,a,z,a',delta
            call allocate_dydx(set_type,conf)
            call allocate_units(set_type,conf)
            
            !* forward prop for energy per atom
            call forward_propagate(set_type,conf)
    
            !* need dy_atm/dx for all atoms and features
            call backward_propagate(set_type,conf)

            if (calculate_property("forces")) then
                call calculate_forces(set_type,conf)
                call calculate_d2ydxdx(set_type,conf)
            end if

            !* (E_ref - \sum_i E_i)^2
            tmpE = sum(data_sets(set_type)%configs(conf)%current_ei) &
                    &-data_sets(set_type)%configs(conf)%ref_energy 

            invN = 1.0d0/dble(data_sets(set_type)%configs(conf)%n)

            if(loss_norm_type.eq.1) then
                tmpE = sign(1.0d0,tmpE) * invN
            else if (loss_norm_type.eq.2) then 
                tmpE = 2.0d0*sign(1.0d0,tmpE)*abs(tmpE) * (invN**2)
            end if
            !* loss constant for energy derivative wrt feature params
            tmpE = tmpE * loss_const_energy


            !* dy_atm / dparam = sum_ft dy_atm/dx_ft * dx_ft/dparam
            do atm=1,data_sets(set_type)%configs(conf)%n,1
                !* atomic number of central atom
                !zatm = feature_isotropic(atm)%z_atom
                zatm = set_neigh_info(conf)%twobody(atm)%z_atom
               
                !if (feature_isotropic(atm)%n.le.0) then
                if (set_neigh_info(conf)%twobody(atm)%n.le.0) then
                    cycle
                end if
                
                !do neigh=1,feature_isotropic(atm)%n,1
                do neigh=1,set_neigh_info(conf)%twobody(atm)%n,1
                    !* atom-atom distance
                    !dr = feature_isotropic(atm)%dr(neigh)
                    dr = set_neigh_info(conf)%twobody(atm)%dr(neigh)

                    !* atomic number
                    !zngh = feature_isotropic(atm)%z(neigh)
                    zngh = set_neigh_info(conf)%twobody(atm)%z(neigh)

                    do ft=1,feature_params%num_features,1
                        ftype = feature_params%info(ft)%ftype
                        
                        if (ftype.eq.featureID_StringToInt("atomic_number")) then
                            cycle
                        else if (ftype.eq.featureID_StringToInt("acsf_behler-g1")) then
                            cycle
                        else if (feature_IsTwoBody(ftype).neqv..true.) then
                            cycle
                        end if
                        
                        !* need to multiply by dy/dx_ft
                        call feature_TwoBody_param_deriv(dr,zatm,zngh,ft,&
                                &dxdparam(atm)%info(ft))

                        if (calculate_property("forces")) then
                            !* d f_atm / d ft_param += - d y_neigh / d ft_neigh * 
                            !* d^2 ft_neigh /  d r_atom d ft_param
                            call feature_TwoBody_param_forces_deriv(conf,atm,neigh,ft,&
                                    &d2xdrdparam)
                        end if
                    end do !* end loop over features
                end do !* end loop over two body neighbours to atm
                    
                if (calculate_property("forces")) then
                    do ft=1,feature_params%num_features,1
                        ftype = feature_params%info(ft)%ftype
                        
                        if (ftype.eq.featureID_StringToInt("atomic_number")) then
                            cycle
                        else if (ftype.eq.featureID_StringToInt("acsf_behler-g1")) then
                            cycle
                        else if (feature_IsTwoBody(ftype)) then
                            !* d x_ft,atm / d r_atm
                            call feature_TwoBody_param_forces_deriv(conf,atm,0,ft,&
                                    &d2xdrdparam)
                        end if
                    end do !* end loop over features
                end if 

                if (calc_threebody) then
                    !if (feature_threebody_info(atm)%n.le.0) then
                    if (set_neigh_info(conf)%threebody(atm)%n.le.0) then
                        cycle
                    end if
                    
                    !do bond=1,feature_threebody_info(atm)%n,1
                    do bond=1,set_neigh_info(conf)%threebody(atm)%n,1
                        do ft=1,feature_params%num_features,1
                            ftype = feature_params%info(ft)%ftype
                            
                            if (ftype.eq.featureID_StringToInt("atomic_number")) then
                                cycle
                            else if (feature_IsTwoBody(ftype)) then
                                cycle
                            end if
                            
                            call feature_ThreeBody_param_deriv(&
                                    &set_neigh_info(conf)%threebody(atm)%dr(1:3,bond),&
                                    &set_neigh_info(conf)%threebody(atm)%cos_ang(bond),&
                                    zatm,set_neigh_info(conf)%threebody(atm)%z(1:2,bond),ft,&
                                    &dxdparam(atm)%info(ft))

                            if (calculate_property("forces")) then
                                call feature_ThreeBody_param_forces_deriv(set_type,conf,atm,&
                                        &neigh,ft,d2xdrdparam)
                            end if 
                        end do !* end loop over features
                    end do !* end loop over threebody bonds to atm
                end if !* end if three body interactions

                !* dy_atm / dparam = sum_ft dy_atm/dx_ft * dx_ft/dparam
                call append_atom_contribution(atm,dxdparam(atm),tmpE,lcl_feat_derivs)

            end do !* end loop over atoms

      
            if (calculate_property("forces")) then
                !* norm between model and ref forces
                do atm=1,data_sets(set_type)%configs(conf)%n
                    do xx=1,3,1
                        force_norm_const(xx,atm) = sign(1.0d0,&
                                &data_sets(set_type)%configs(conf)%current_fi(xx,atm)-&
                                &data_sets(set_type)%configs(conf)%ref_fi(xx,atm))
                        if (loss_norm_type.eq.2) then
                            force_norm_const(xx,atm) = 2.0d0*force_norm_const(xx,atm)*abs(&
                                    &data_sets(set_type)%configs(conf)%current_fi(xx,atm)-&
                                    &data_sets(set_type)%configs(conf)%ref_fi(xx,atm))
                        end if 
                    end do !* end loop over cartesian components
                end do !* end loop over local atoms
                
                call calculate_dfdparam(set_type,conf,force_norm_const,dxdparam,d2xdrdparam,&
                &force_contribution)
                 
                !* append force contribution to loss derivative 
                call add_feat_param_derivs(lcl_feat_derivs,force_contribution,loss_const_forces)
            end if
           
            if (allocated(ultra_z)) then 
                deallocate(ultra_z)
                deallocate(ultra_idx)
                deallocate(ultra_cart)
            end if
        end subroutine single_conf_feat_jac

        subroutine feature_TwoBody_param_deriv(dr,zatm,zngh,ft_idx,feature_deriv)
            implicit none

            real(8),intent(in) :: dr,zatm,zngh
            integer,intent(in) :: ft_idx
            type(feature_),intent(inout) :: feature_deriv

            !* scratch 
            integer :: ftype,kk,fourier_terms
            real(8) :: mu,prec,scl_cnst,fs,rcut
            real(8) :: tmp1,tmp2,tmpz,za,zb,tmp_taper,tmp_cnst

            ftype = feature_params%info(ft_idx)%ftype

            !* constant shift due to original pre conditioning
            scl_cnst = feature_params%info(ft_idx)%scl_cnst

            !* atomic number contribution
            za = feature_params%info(ft_idx)%za
            zb = feature_params%info(ft_idx)%zb
            tmpz = (zatm+1.0d0)**za * (zngh+1.0d0)**zb

            !* tapering
            rcut = feature_params%info(ft_idx)%rcut
            fs = feature_params%info(ft_idx)%fs
            tmp_taper = taper_1(dr,rcut,fs)

            !* constant term
            tmp_cnst = scl_cnst*tmpz*tmp_taper

            if (ftype.eq.featureID_StringToInt("acsf_behler-g2")) then
                !* exponential apart from missing 1/2 factor in exp.
                mu = feature_params%info(ft_idx)%rs
                prec = feature_params%info(ft_idx)%eta
           
                tmp1 = dr-mu
                tmp2 = -(tmp1**2)
            
                feature_deriv%rs = feature_deriv%rs + 2.0d0*tmp_cnst*tmp1*prec*&
                        &exp(prec*tmp2)

                feature_deriv%eta = feature_deriv%eta + tmp_cnst*tmp2*&
                        &exp(prec*tmp2)
                
            else if (ftype.eq.featureID_StringToInt("acsf_normal-b2")) then
                !* params
                mu = feature_params%info(ft_idx)%mean(1)
                prec = feature_params%info(ft_idx)%prec(1,1)
           
                tmp1 = dr-mu
                tmp2 = -0.5d0*(tmp1**2)
            
                feature_deriv%mean(1) = feature_deriv%mean(1) + tmp_cnst*tmp1*prec*&
                        &exp(prec*tmp2)

                feature_deriv%prec(1,1) = feature_deriv%prec(1,1) + tmp_cnst*tmp2*&
                        &exp(prec*tmp2)

            else if (ftype.eq.featureID_StringToInt("acsf_fourier-b2")) then
                !* 2 pi r/rcut
                tmp1 = dr*6.28318530718d0 / rcut

                fourier_terms = int(size(feature_params%info(ft_idx)%linear_w)/2)

                do kk=1,fourier_terms,1
                    feature_deriv%linear_w(kk) = feature_deriv%linear_w(kk) + &
                            &sin(dble(kk)*tmp1)*tmp_taper
                    
                    feature_deriv%linear_w(kk+fourier_terms) = &
                            &feature_deriv%linear_w(kk+fourier_terms) + cos(dble(kk)*tmp1)*tmp_taper
                end do
            else
                call error("feature_TwoBody_param_deriv","Implementation error")
            end if
        end subroutine feature_TwoBody_param_deriv
       
        subroutine feature_ThreeBody_param_deriv(dr_array,cos_ang,zatm,z_array,ft_idx,&
        &lcl_feat_deriv)
            implicit none

            real(8),intent(in) :: dr_array(1:3),cos_ang
            real(8),intent(in) :: zatm,z_array(1:2)
            integer,intent(in) :: ft_idx
            type(feature_),intent(inout) :: lcl_feat_deriv

            !* scratch
            real(8) :: drij,drik,drjk,rcut,scl_cnst
            real(8) :: tap_ij,tap_ik,tap_jk,tmp_z,za,zb
            real(8) :: lambda,eta,xi,tmp_1,tmp_2,tmp_3
            real(8) :: tmp_4,mean(1:3),prec(1:3,1:3)
            real(8) :: lwork_1(1:3),lwork_2(1:3),perm_1(1:3)
            real(8) :: perm_2(1:3),exp_1,exp_2,fs
            real(8) :: zj,zk,const
            integer :: ftype,ii,jj

            drij = dr_array(1)
            drik = dr_array(2)
            drjk = dr_array(3)
            zj = z_array(1)
            zk = z_array(2)

            scl_cnst = feature_params%info(ft_idx)%scl_cnst
            rcut = feature_params%info(ft_idx)%rcut
            fs = feature_params%info(ft_idx)%fs
            za = feature_params%info(ft_idx)%za
            zb = feature_params%info(ft_idx)%zb
            
            !* tapering
            tap_ij = taper_1(drij,rcut,fs)
            tap_ik = taper_1(drik,rcut,fs)
            tap_jk = taper_1(drjk,rcut,fs)

            !* atomic number contribution
            tmp_z = (zatm+1.0d0)**za * ((zj+1.0d0)*(zk+1.0d0))**zb

            ftype = feature_params%info(ft_idx)%ftype
            
            if ( (ftype.eq.featureID_StringToInt("acsf_behler-g4")).or.&
            &(ftype.eq.featureID_StringToInt("acsf_behler-g5")) ) then
                
                if (ftype.eq.featureID_StringToInt("acsf_behler-g4")) then
                    tmp_3 = scl_cnst*tap_ij*tap_ik*tap_jk*tmp_z
                    tmp_4 = drij**2+drik**2+drjk**2
                else if (ftype.eq.featureID_StringToInt("acsf_behler-g5")) then
                    tmp_3 = scl_cnst*tap_ij*tap_ik*tmp_z
                    tmp_4 = drij**2+drik**2
                end if



                xi = feature_params%info(ft_idx)%xi
                lambda = feature_params%info(ft_idx)%lambda
                eta = feature_params%info(ft_idx)%eta
                
                tmp_1 = 2.0d0**(1.0d0-xi)*(1.0d0+lambda*cos_ang)**xi
                tmp_2 = exp(-eta*tmp_4)

                lcl_feat_deriv%xi = lcl_feat_deriv%xi + (log(1.0d0+lambda*cos_ang) - log(2.0d0))*&
                        &tmp_1*tmp_2*tmp_3

                lcl_feat_deriv%eta = lcl_feat_deriv%eta - tmp_1*tmp_2*tmp_3*tmp_4
            
            else if (ftype.eq.featureID_StringToInt("acsf_normal-b3")) then
                if ((dr_array(1).gt.rcut).or.(dr_array(2).gt.rcut)) then
                    !* for performance
                    return
                end if

                tmp_3 = scl_cnst*tap_ij*tap_ik*tap_jk*tmp_z
                
                mean = feature_params%info(ft_idx)%mean
                prec = feature_params%info(ft_idx)%prec

                perm_1(1:2) = dr_array(1:2)
                perm_1(3) = cos_ang
                perm_2(1) = dr_array(2)
                perm_2(2) = dr_array(1)
                perm_2(3) = cos_ang

                !* lwork_1 = precision * (mean - x1)
                call dsymv('u',3,1.0d0,prec,3,mean-perm_1,1,0.0d0,lwork_1,1)
                call dsymv('u',3,1.0d0,prec,3,mean-perm_2,1,0.0d0,lwork_2,1)
          
                exp_1 = exp(-0.5d0*ddot(3,mean-perm_1,1,lwork_1,1))
                exp_2 = exp(-0.5d0*ddot(3,mean-perm_2,1,lwork_2,1))
           

                lcl_feat_deriv%mean = lcl_feat_deriv%mean - (lwork_1*exp_1 + lwork_2*exp_2)*tmp_3

                !* d /dLambda_a where a is an off diagonal is really d / dLambda_ij + d / dLambda_ji
                !* ie need to double contribution

                do ii=1,3,1
                    do jj=1,3,1
                        if(ii.eq.jj) then
                            const = 0.5d0
                        else
                            !* for off diagonal contribution, we're really consdering summation of
                            !* partial differentials of transpose element pairs
                            const = 1.0d0
                        end if

                        lcl_feat_deriv%prec(ii,jj) = lcl_feat_deriv%prec(ii,jj) - (exp_1*&
                                &(mean(ii)-perm_1(ii))*(mean(jj)-perm_1(jj)) +exp_2*&
                                &(mean(ii)-perm_2(ii))*(mean(jj)-perm_2(jj))  )*tmp_3*const
                    end do
                end do
            else
                call error("feature_ThreeBody_param_deriv","Implementation error")
            end if

        end subroutine feature_ThreeBody_param_deriv        

        subroutine init_feature_array(lcl_feat_derivs)
            implicit none

            !* args
            type(feature_info),intent(inout) :: lcl_feat_derivs

            !* scratch
            integer :: ft,ftype,num_weights

            if (allocated(lcl_feat_derivs%info)) then
                deallocate(lcl_feat_derivs%info)
            end if
            allocate(lcl_feat_derivs%info(feature_params%num_features))

            lcl_feat_derivs%num_features = feature_params%num_features

            do ft=1,feature_params%num_features,1
                ftype = feature_params%info(ft)%ftype
                lcl_feat_derivs%info(ft)%ftype = ftype

                if (ftype.eq.featureID_StringToInt("acsf_normal-b2")) then
                    allocate(lcl_feat_derivs%info(ft)%mean(1))
                    allocate(lcl_feat_derivs%info(ft)%prec(1,1))
                
                    lcl_feat_derivs%info(ft)%mean(:) = 0.0d0
                    lcl_feat_derivs%info(ft)%prec(:,:) = 0.0d0
                else if (ftype.eq.featureID_StringToInt("acsf_normal-b3")) then
                    allocate(lcl_feat_derivs%info(ft)%mean(3))
                    allocate(lcl_feat_derivs%info(ft)%prec(3,3))
                    
                    lcl_feat_derivs%info(ft)%mean(:) = 0.0d0
                    lcl_feat_derivs%info(ft)%prec(:,:) = 0.0d0
                else if (ftype.eq.featureID_StringToInt("acsf_fourier-b2")) then
                    num_weights = size(feature_params%info(ft)%linear_w)
                    allocate(lcl_feat_derivs%info(ft)%linear_w(num_weights))

                    lcl_feat_derivs%info(ft)%linear_w(:) = 0.0d0
                end if
                
                lcl_feat_derivs%info(ft)%rs = 0.0d0
                lcl_feat_derivs%info(ft)%xi = 0.0d0
                lcl_feat_derivs%info(ft)%eta = 0.0d0
            end do !* end loop over features
        end subroutine init_feature_array

        subroutine zero_feature_info(feature_info_inst)
            implicit none

            type(feature_info),intent(inout) :: feature_info_inst

            !* scratch 
            integer :: ft

            do ft=1,feature_params%num_features,1
                !* all feature attributes that are optimizable
                feature_info_inst%info(ft)%rs = 0.0d0
                feature_info_inst%info(ft)%eta = 0.0d0
                feature_info_inst%info(ft)%xi = 0.0d0
                if (allocated(feature_info_inst%info(ft)%mean)) then
                    !* normal feat. type
                    feature_info_inst%info(ft)%mean = 0.0d0
                    feature_info_inst%info(ft)%prec = 0.0d0
                end if
                if (allocated(feature_info_inst%info(ft)%linear_w)) then
                    feature_info_inst%info(ft)%linear_w = 0.0d0
                end if
            end do
        end subroutine zero_feature_info

        subroutine add_individual_features(feat_in,const,feat_out)
            !* feat_out += feat_in*const
            
            implicit none

            !* args
            real(8),intent(in) :: const
            type(feature_),intent(in) :: feat_in
            type(feature_),intent(inout) :: feat_out
            
            !* scratch 
            integer :: ftype
            
            ftype = feat_in%ftype

            if (ftype.eq.featureID_StringToInt("acsf_behler-g2")) then
                feat_out%eta = feat_out%eta + feat_in%eta*const
                feat_out%rs  = feat_out%rs  + feat_in%rs*const
            else if ( (ftype.eq.featureID_StringToInt("acsf_behler-g4")).or.&
            &(ftype.eq.featureID_StringToInt("acsf_behler-g5")) ) then
                feat_out%eta = feat_out%eta + feat_in%eta*const
                feat_out%xi  = feat_out%xi  + feat_in%xi*const
            else if ( (ftype.eq.featureID_StringToInt("acsf_normal-b2")).or.&
            &(ftype.eq.featureID_StringToInt("acsf_normal-b3")) ) then
                feat_out%mean = feat_out%mean + feat_in%mean*const
                feat_out%prec = feat_out%prec + feat_in%prec*const
            else if (ftype.eq.featureID_StringToInt("acsf_fourier-b2")) then
                feat_out%linear_w = feat_out%linear_w + feat_in%linear_w*const
            end if
        end subroutine add_individual_features

        subroutine add_feat_param_derivs(gbl_feat_derivs,lcl_feat_derivs,const)
            !* add tmp_feat_derivs*const to lcl_feat_derivs
            
            implicit none

            type(feature_info),intent(inout) :: gbl_feat_derivs
            type(feature_info),intent(in) :: lcl_feat_derivs
            real(8),intent(in) :: const

            !* scratch
            integer :: ft

            do ft=1,feature_params%num_features,1
                call add_individual_features(lcl_feat_derivs%info(ft),const,&
                        &gbl_feat_derivs%info(ft))
            end do !* end loop over features
        end subroutine
                
        subroutine append_atom_contribution(atm,tmp_feat_derivs,tmpE,lcl_feat_derivs)
            implicit none

            integer,intent(in) :: atm
            type(feature_info),intent(in) :: tmp_feat_derivs
            real(8),intent(in) :: tmpE
            type(feature_info),intent(inout) :: lcl_feat_derivs
            
            !* scratch
            integer :: ft,ftype

            do ft=1,feature_params%num_features,1
                !* dydx(ft,atm) * dx/dparam

                ftype = feature_params%info(ft)%ftype

                if (ftype.eq.featureID_StringToInt("acsf_behler-g2")) then
                    lcl_feat_derivs%info(ft)%eta = lcl_feat_derivs%info(ft)%eta + &
                            &tmp_feat_derivs%info(ft)%eta*tmpE * dydx(ft,atm)
                    
                    lcl_feat_derivs%info(ft)%rs = lcl_feat_derivs%info(ft)%rs + &
                            &tmp_feat_derivs%info(ft)%rs*tmpE * dydx(ft,atm)
                
                else if ( (ftype.eq.featureID_StringToInt("acsf_behler-g4")).or.&
                &(ftype.eq.featureID_StringToInt("acsf_behler-g5")) ) then
                    lcl_feat_derivs%info(ft)%eta = lcl_feat_derivs%info(ft)%eta + &
                            &tmp_feat_derivs%info(ft)%eta*tmpE * dydx(ft,atm)
                    
                    lcl_feat_derivs%info(ft)%xi = lcl_feat_derivs%info(ft)%xi + &
                            &tmp_feat_derivs%info(ft)%xi*tmpE * dydx(ft,atm)
                
                else if ( (ftype.eq.featureID_StringToInt("acsf_normal-b2")).or.& 
                &(ftype.eq.featureID_StringToInt("acsf_normal-b3")) ) then
                    lcl_feat_derivs%info(ft)%mean = lcl_feat_derivs%info(ft)%mean + &
                            &tmp_feat_derivs%info(ft)%mean*tmpE * dydx(ft,atm)
                    
                    lcl_feat_derivs%info(ft)%prec = lcl_feat_derivs%info(ft)%prec + &
                            &tmp_feat_derivs%info(ft)%prec*tmpE * dydx(ft,atm)
                
                else if (ftype.eq.featureID_StringToInt("acsf_fourier-b2")) then
                    lcl_feat_derivs%info(ft)%linear_w = lcl_feat_derivs%info(ft)%linear_w + &
                            &tmp_feat_derivs%info(ft)%linear_w*tmpE * dydx(ft,atm)
                end if
            end do
        end subroutine append_atom_contribution


        subroutine parse_feature_format_to_array_jac(gbl_feat_derivs,jac_array)
            implicit none

            !* args
            type(feature_info),intent(in) :: gbl_feat_derivs
            real(8),intent(inout) :: jac_array(:)

            !* scratch
            integer :: ft,ftype,cntr,ii,jj

            cntr = 0
            jac_array = 0.0d0

            do ft=1,feature_params%num_features,1
                ftype = gbl_feat_derivs%info(ft)%ftype

                if (ftype.eq.featureID_StringToInt("atomic_number")) then
                    cycle
                else if (ftype.eq.featureID_StringToInt("acsf_behler-g2")) then
                    call update_array_idx(cntr,1,size(jac_array))
                    jac_array(cntr) = gbl_feat_derivs%info(ft)%eta
                    
                    call update_array_idx(cntr,1,size(jac_array))
                    jac_array(cntr) = gbl_feat_derivs%info(ft)%rs
                else if ( (ftype.eq.featureID_StringToInt("acsf_behler-g4")).or.&
                &(ftype.eq.featureID_StringToInt("acsf_behler-g5")) ) then
                    call update_array_idx(cntr,1,size(jac_array))
                    jac_array(cntr) = gbl_feat_derivs%info(ft)%xi
                    
                    call update_array_idx(cntr,1,size(jac_array))
                    jac_array(cntr) = gbl_feat_derivs%info(ft)%eta
                else if (ftype.eq.featureID_StringToInt("acsf_normal-b2")) then
                    call update_array_idx(cntr,1,size(jac_array))
                    jac_array(cntr) = gbl_feat_derivs%info(ft)%prec(1,1)
                    
                    call update_array_idx(cntr,1,size(jac_array))
                    jac_array(cntr) = gbl_feat_derivs%info(ft)%mean(1)
                else if (ftype.eq.featureID_StringToInt("acsf_normal-b3")) then
                    do ii=1,3,1
                        do jj=ii,3,1
                            call update_array_idx(cntr,1,size(jac_array))
                            jac_array(cntr) = gbl_feat_derivs%info(ft)%prec(ii,jj)
                        end do
                    end do !* end loop over upper triangular matrix
                    call update_array_idx(cntr,1,size(jac_array))
                    jac_array(cntr:cntr+2) = gbl_feat_derivs%info(ft)%mean(1:3)
                    call update_array_idx(cntr,2,size(jac_array))
                else if (ftype.eq.featureID_StringToInt("acsf_fourier-b2")) then
                    do ii=1,size(gbl_feat_derivs%info(ft)%linear_w),1
                        call update_array_idx(cntr,1,size(jac_array))
                        jac_array(cntr) = gbl_feat_derivs%info(ft)%linear_w(ii)
                    end do !* end loop over fourier weights
                end if
            end do !* end loop over feature types
        end subroutine parse_feature_format_to_array_jac

        subroutine update_array_idx(current_val,increment,maxvalue)
            implicit none
            
            integer,intent(inout) :: current_val
            integer,intent(in) :: maxvalue,increment

            if (current_val+increment.gt.maxvalue) then
                call error("update_array_idx","Error parsing feature format into array")
            else
                current_val = current_val + increment
            end if
        end subroutine update_array_idx

        integer function num_optimizable_params()
            implicit none

            integer :: cntr,ft,ftype

            cntr = 0

            do ft=1,feature_params%num_features,1
                ftype = feature_params%info(ft)%ftype

                if (ftype.eq.featureID_StringToInt("acsf_behler-g2")) then
                    cntr = cntr + 2
                else if (ftype.eq.featureID_StringToInt("acsf_behler-g4")) then
                    cntr = cntr + 2
                else if (ftype.eq.featureID_StringToInt("acsf_behler-g5")) then
                    cntr = cntr + 2
                else if (ftype.eq.featureID_StringToInt("acsf_normal-b2")) then
                    cntr = cntr + 2
                else if (ftype.eq.featureID_StringToInt("acsf_normal-b3")) then
                    cntr = cntr + 9
                else if (ftype.eq.featureID_StringToInt("acsf_fourier-b2")) then
                    cntr = cntr + size(feature_params%info(ft)%linear_w)
                end if
            end do
            num_optimizable_params = cntr
        end function num_optimizable_params
    
        subroutine feature_TwoBody_param_forces_deriv(conf,atm,neigh,ft_idx,&
        &d2xdrdparam)
            implicit none

            integer,intent(in) :: conf,atm,neigh,ft_idx
            type(feature_info),intent(inout) :: d2xdrdparam(:,:,:)

            !* scratch
            real(8) :: rcut,tmp1,scl_cnst,za,zb,tmpz,tmp_taper,tmp_taper_deriv
            real(8) :: tmp_cnst,rs,eta,tmp2,tmp3,tmp4,tmp5,dr_vec(1:3)
            real(8) :: dr,fs,zatm,zngh,mu,lambda,kk_dble,tmps,tmpc
            integer :: ii,kk,ftype,neigh_idx,lim1,lim2,deriv_idx,xx
            integer :: fourier_terms

            if (neigh.eq.0) then
                !* take derivative wrt central atom (atm)
                lim1 = 1
                lim2 = set_neigh_info(conf)%twobody(atm)%n
                deriv_idx = atm
            else
                lim1 = neigh
                lim2 = neigh
                deriv_idx = set_neigh_info(conf)%twobody(atm)%idx(neigh)
            end if

            do ii=lim1,lim2,1
                ! NO SELF INTERACTION
                if (atm.eq.set_neigh_info(conf)%twobody(atm)%idx(ii)) then
                    !* d r_i - (r_i + const) / d r_i = 0
                    cycle
                end if


                !* neighbour info
                dr = set_neigh_info(conf)%twobody(atm)%dr(ii)
                neigh_idx = set_neigh_info(conf)%twobody(atm)%idx(ii)
                zatm = set_neigh_info(conf)%twobody(atm)%z_atom
                zngh = set_neigh_info(conf)%twobody(atm)%z(ii)

                if (neigh.eq.0) then
                    !* d rij / d r_central
                    dr_vec(:) = -set_neigh_info(conf)%twobody(atm)%drdri(:,ii)
                else
                    !* d rij / d r_neighbour
                    dr_vec(:) = set_neigh_info(conf)%twobody(atm)%drdri(:,ii)
                end if

                
                ftype = feature_params%info(ft_idx)%ftype

                !* constant shift due to original pre conditioning
                scl_cnst = feature_params%info(ft_idx)%scl_cnst

                !* atomic number contribution
                za = feature_params%info(ft_idx)%za
                zb = feature_params%info(ft_idx)%zb
                tmpz = (zatm+1.0d0)**za * (zngh+1.0d0)**zb

                !* tapering
                rcut = feature_params%info(ft_idx)%rcut
                fs = feature_params%info(ft_idx)%fs

! debug
if (dr.gt.rcut) then
    write(*,*) 'dr > rcut'
    call exit(0)
end if
! debug

                if (speedup_applies("twobody_rcut")) then
                    tmp_taper = set_neigh_info(conf)%twobody(atm)%dr_taper(ii)
                    tmp_taper_deriv = set_neigh_info(conf)%twobody(atm)%dr_taper_deriv(ii)
                else
                    tmp_taper = taper_1(dr,rcut,fs)
                    tmp_taper_deriv = taper_deriv_1(dr,rcut,fs)
                end if

                !* constant term
                tmp_cnst = scl_cnst*tmpz
                
                if (ftype.eq.featureID_StringToInt("acsf_behler-g2")) then
                    !* params
                    eta = feature_params%info(ft_idx)%eta
                    rs = feature_params%info(ft_idx)%rs
                    
                    tmp1 = dr - rs
                    tmp2 = exp(-eta*(tmp1**2))
                    
                    !* d x / d rij
                    tmp3 = tmp2 * (tmp_taper_deriv - 2.0d0*tmp_taper*eta*tmp1)
                   
                    !* d^2 / d eta 
                    tmp4 = -tmp1*(tmp1*tmp3 + 2.0d0*tmp_taper*tmp2)

                    !* d^2 / d rs
                    tmp5 = 2.0d0*eta*( tmp1*tmp3 + tmp_taper*tmp2 )

                    !* d x / d eta
                    !tmp6 = -(tmp1**2)*tmp2*tmp_taper
                    
                    !* d x / d rs
                    !tmp7 = 2.0d0*eta*tmp1*tmp2*tmp_taper

                    do xx=1,3
                        d2xdrdparam(deriv_idx,atm,xx)%info(ft_idx)%eta = &
                                &d2xdrdparam(deriv_idx,atm,xx)%info(ft_idx)%eta + &
                                &tmp_cnst*tmp4*dr_vec(xx)
                        
                        d2xdrdparam(deriv_idx,atm,xx)%info(ft_idx)%rs = &
                                &d2xdrdparam(deriv_idx,atm,xx)%info(ft_idx)%rs + &
                                &tmp_cnst*tmp5*dr_vec(xx)
                    end do

                else if (ftype.eq.featureID_StringToInt("acsf_normal-b2")) then
                    !* params
                    mu = feature_params%info(ft_idx)%mean(1)
                    lambda = feature_params%info(ft_idx)%prec(1,1)

                    tmp1 = dr - mu
                    tmp2 = exp(-0.5d0*(dr-mu)**2 * lambda)

                    !* d x / d rij
                    tmp3 = tmp2*(tmp_taper_deriv - tmp1*lambda*tmp_taper)
                    
                    !* d^2 / d precision
                    tmp4 = -tmp1*(0.5d0*tmp1*tmp3 + tmp2*tmp_taper)
                    
                    !* d^2 / d mean
                    tmp5 = lambda*(tmp1*tmp3 + tmp2*tmp_taper)
                    
                    do xx=1,3
                        d2xdrdparam(deriv_idx,atm,xx)%info(ft_idx)%prec(1,1) = &
                                &d2xdrdparam(deriv_idx,atm,xx)%info(ft_idx)%prec(1,1) + &
                                &tmp_cnst*tmp4*dr_vec(xx)
                        
                        d2xdrdparam(deriv_idx,atm,xx)%info(ft_idx)%mean(1) = &
                                &d2xdrdparam(deriv_idx,atm,xx)%info(ft_idx)%mean(1) + &
                                &tmp_cnst*tmp5*dr_vec(xx)
                    end do
                else if (ftype.eq.featureID_StringToInt("acsf_fourier-b2")) then
                    !* 2 pi / rcut
                    tmp1 = 6.28318530718d0/rcut

                    !* 2 pi r / rcut
                    tmp2 = tmp1*dr
                    
                    fourier_terms = int(size(feature_params%info(ft_idx)%linear_w)/2)


                    do xx=1,3
                        do kk=1,fourier_terms,1
                            tmps = sin(tmp2*kk_dble)
                            tmpc = cos(tmp2*kk_dble)
                            
                            d2xdrdparam(deriv_idx,atm,xx)%info(ft_idx)%linear_w(kk) = &
                                    &d2xdrdparam(deriv_idx,atm,xx)%info(ft_idx)%linear_w(kk) +&
                                    &( tmp_taper_deriv*tmps + tmp_taper*tmp1*kk_dble*tmpc )*&
                                    &dr_vec(xx)
                            
                            d2xdrdparam(deriv_idx,atm,xx)%info(ft_idx)%linear_w(kk+fourier_terms) = &
                                    &d2xdrdparam(deriv_idx,atm,xx)%info(ft_idx)%&
                                    &linear_w(kk+fourier_terms) +&
                                    &( tmp_taper_deriv*tmpc - tmp_taper*tmp1*kk_dble*tmps )*&
                                    &dr_vec(xx)

                        end do
                    end do
                else
                    call error("feature_TwoBody_param_forces_deriv","Implementation error")
                end if

            end do !* end loop over ii (neighbours to take derivative wrt)

        end subroutine feature_TwoBody_param_forces_deriv
        
        
        subroutine feature_ThreeBody_param_forces_deriv(set_type,conf,atm,bond_idx,ft_idx,&
        &d2xdrdparam)
            implicit none
            
            !* args
            integer,intent(in) :: set_type,conf,atm,bond_idx,ft_idx
            type(feature_info),intent(inout) :: d2xdrdparam(:,:,:)

            !* scratch
            real(8) :: rcut,fs,za,zb,lambda,eta,xi,scl
            real(8) :: drij,drik,drjk,tap_ij,tap_ik,tap_jk
            real(8) :: tap_ij_deriv,tap_ik_deriv,tap_jk_deriv
            real(8) :: tmp_z,drijdrz(1:3),drikdrz(1:3),drjkdrz(1:3)
            real(8) :: const(1:8),cos_angle,dcosdrz(1:3)
            integer :: deriv_idx,ftype,zz,xx

            rcut   = feature_params%info(ft_idx)%rcut
            fs     = feature_params%info(ft_idx)%fs
            za     = feature_params%info(ft_idx)%za
            zb     = feature_params%info(ft_idx)%zb
            ftype  = feature_params%info(ft_idx)%ftype
            scl    = feature_params%info(ft_idx)%scl_cnst

            drij = set_neigh_info(conf)%threebody(atm)%dr(1,bond_idx)
            drik = set_neigh_info(conf)%threebody(atm)%dr(2,bond_idx)
            drjk = set_neigh_info(conf)%threebody(atm)%dr(3,bond_idx)

            if ( (drij.gt.rcut).or.(drik.gt.rcut).or.(drjk.gt.rcut) ) then
                return
            end if

            cos_angle = set_neigh_info(conf)%threebody(atm)%cos_ang(bond_idx)

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
   
            !* atomic contribution and unit rescaling
            tmp_z = ( ((set_neigh_info(conf)%threebody(atm)%z(1,bond_idx)+1.0d0)*&
                    &(set_neigh_info(conf)%threebody(atm)%z(2,bond_idx)+1.0d0))**zb *&
                    &(set_neigh_info(conf)%threebody(atm)%z_atom+1.0d0)**za ) * scl

            !* 1= deriv wrt. jj, 2= deriv wrt. kk, 3= deriv wrt. ii
            do zz=1,3,1
                !* d^2 x_atom / dr_deriv_idx dparam
                if (zz.lt.3) then
                    deriv_idx = set_neigh_info(conf)%threebody(atm)%idx(zz,bond_idx) 
                else
                    deriv_idx = atm
                end if

                !* angle derivative wrt zz
                dcosdrz =  set_neigh_info(conf)%threebody(atm)%dcos_dr(:,zz,bond_idx)

                !* distance derivative wrt zz
                if (zz.eq.1) then
                    !* deriv wrt jj
                    drijdrz =  set_neigh_info(conf)%threebody(atm)%drdri(:,1,bond_idx)
                    drikdrz =  set_neigh_info(conf)%threebody(atm)%drdri(:,4,bond_idx)
                    drjkdrz = -set_neigh_info(conf)%threebody(atm)%drdri(:,5,bond_idx)
                else if (zz.eq.2) then
                    !* deriv wrt kk
                    drijdrz =  set_neigh_info(conf)%threebody(atm)%drdri(:,2,bond_idx)
                    drikdrz =  set_neigh_info(conf)%threebody(atm)%drdri(:,3,bond_idx)
                    drjkdrz =  set_neigh_info(conf)%threebody(atm)%drdri(:,5,bond_idx)
                else if (zz.eq.3) then
                    !* deriv wrt ii (central atom)
                    drijdrz = -set_neigh_info(conf)%threebody(atm)%drdri(:,1,bond_idx)
                    drikdrz = -set_neigh_info(conf)%threebody(atm)%drdri(:,3,bond_idx)
                    drjkdrz =  set_neigh_info(conf)%threebody(atm)%drdri(:,6,bond_idx)
                end if

                if (ftype.eq.featureID_StringToInt("acsf_behler-g4")) then
                    !* feature specific params
                    eta    = feature_params%info(ft_idx)%eta 
                    lambda = feature_params%info(ft_idx)%lambda 
                    xi     = feature_params%info(ft_idx)%xi  
                    
                    !* exp(-eta*(drij^2 + drik^2 + drjk^2))
                    const(1) = exp(-eta*(drij**2 + drik**2 + drjk**2))

                    !* 2 ^(1-xi)
                    const(2) = 2.0d0**(1.0d0-xi)

                    !* ( 1 + lambda*cos(theta) )
                    const(3) = (1.0d0 + lambda*cos_angle)

                    !* ( 1 + lambda*cos(theta) )^xi
                    const(4) = const(3)**xi

                    const(5) = tap_ij*tap_ik*tap_jk*lambda*product(const(1:2))*&
                            &const(3)**(xi-2.0d0) * (xi*(xi-1.0d0) + const(3)*(xi*(1.0d0-xi)*0.5d0+&
                            &1.0d0))
                            

                    !* (1-xi)*(2^xi)* (1+lambda*cos(theta))^xi  + 
                    !* 2^(1-xi) * xi *(1+lambda*cos(theta))^(xi-1) * exp(-eta ... ) * xi
                    const(6) = const(1)*const(2)*(const(3)**(xi-1.0d0)) *&
                            &(xi + 0.5d0*(1.0d0-xi)*const(3))

                    !* tapering product
                    const(7) = tap_ij*tap_ik*tap_jk

                    !* summation of distances squared
                    const(8) = -(drij**2 + drik**2 + drjk**2)

                    do xx=1,3,1
                        d2xdrdparam(deriv_idx,atm,xx)%info(ft_idx)%xi = &
                                &d2xdrdparam(deriv_idx,atm,xx)%info(ft_idx)%xi + &
                                &(  const(5)*dcosdrz(xx) + &
                                &(tap_ik*tap_jk*(tap_ij_deriv-2.0d0*eta*tap_ij*drij)*drijdrz(xx)+&
                                & tap_ij*tap_jk*(tap_ik_deriv-2.0d0*eta*tap_ik*drik)*drikdrz(xx)+&
                                & tap_ij*tap_ik*(tap_jk_deriv-2.0d0*eta*tap_jk*drjk)*drjkdrz(xx))*&
                                &const(6)  ) * tmp_z

                        d2xdrdparam(deriv_idx,atm,xx)%info(ft_idx)%eta = & 
                                &d2xdrdparam(deriv_idx,atm,xx)%info(ft_idx)%eta + &
                                &tmp_z*const(2)*(const(3)**(xi-1.0d0))*const(1)*(&
                                &xi*lambda*const(7)*const(8)*dcosdrz(xx) + &
                                const(3)*(&
                                &tap_ij*tap_ik*drjkdrz(xx)*( (tap_jk_deriv-2.0d0*eta*tap_jk*drjk)*&
                                &const(8) - 2.0d0*tap_jk*drjk) + &
                                &tap_ij*tap_jk*drikdrz(xx)*( (tap_ik_deriv-2.0d0*eta*tap_ik*drik)*&
                                &const(8) - 2.0d0*tap_ik*drik) + &
                                &tap_ik*tap_jk*drijdrz(xx)*( (tap_ij_deriv-2.0d0*eta*tap_ij*drij)*&
                                &const(8) - 2.0d0*tap_ij*drij) )  )
                    end do

                else
                    call error("feature_ThreeBody_param_deriv","Implementation error")
                end if
            end do
        end subroutine feature_ThreeBody_param_forces_deriv
      
        subroutine calculate_dfdparam(set_type,conf,norm_consts,dxdparam,d2xdrdparam,&
        &force_contribution)
            implicit none

            !* args
            integer,intent(in) :: set_type,conf
            real(8),intent(in) :: norm_consts(1:3,1:data_sets(set_type)%configs(conf)%n)
            type(feature_info),intent(in) :: dxdparam(1:data_sets(set_type)%configs(conf)%n)
            type(feature_info),intent(in) :: d2xdrdparam(1:data_sets(set_type)%configs(conf)%n,&
                                                        &1:data_sets(set_type)%configs(conf)%n,1:3)
            type(feature_info),intent(inout) :: force_contribution

            !* scratch
            integer :: ii,jj,xx,neigh_idx,ft,ft2
            real(8) :: const

            do jj=1,data_sets(set_type)%configs(conf)%n,1
                do ft=1,feature_params%num_features,1
                    if (data_sets(set_type)%configs(conf)%x_deriv(ft,jj)%n.le.0) then
                        cycle
                    end if
                    
                    do neigh_idx=1,data_sets(set_type)%configs(conf)%x_deriv(ft,jj)%n,1
                        ii = data_sets(set_type)%configs(conf)%x_deriv(ft,jj)%idx(neigh_idx)

                        do ft2=1,feature_params%num_features,1
                            !* need to loop over non-diagonal elements of hessian
                            do xx=1,3
                                const = -d2ydxdx(ft2,ft,jj)*norm_consts(xx,ii)*data_sets(set_type)%&
                                        &configs(conf)%x_deriv(ft,jj)%vec(xx,neigh_idx)

                                call add_individual_features(dxdparam(jj)%info(ft2),const,&
                                        &force_contribution%info(ft2))
                            end do
                        end do !* end loop over 2nd features

                        do xx=1,3
                            const = -dydx(ft,jj)*norm_consts(xx,ii)

                            call add_individual_features(d2xdrdparam(ii,jj,xx)%info(ft),&
                                    &const,force_contribution%info(ft))
                        end do !* end loop over cartesian components
                    end do !* end loop over neighbours to atm
                end do !* end loop over features
            end do !* end loop over local atoms

        end subroutine calculate_dfdparam

end module feature_selection
