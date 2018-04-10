module feature_selection
    use feature_util, only : get_ultracell, maxrcut, threebody_features_present
    use feature_util, only : calculate_threebody_info,calculate_twobody_info
    use tapering, only : taper_1
    use propagate, only : forward_propagate,backward_propagate
    use features, only : calculate_all_features
    use io, only : error
    use init, only : allocate_units
    use util, only : parse_array_to_structure,copy_weights_to_nobiasT
    use util, only : allocate_dydx
    use feature_config
    use config

    implicit none

    real(8),external :: ddot

    contains
        subroutine loss_feature_jacobian(flat_weights,set_type,parallel,jacobian)
            implicit none

            real(8),intent(in) :: flat_weights(:)
            integer,intent(in) :: set_type
            logical,intent(in) :: parallel
            real(8),intent(inout) :: jacobian(:)

            !* scratch
            integer :: conf
            type(feature_info) :: gbl_derivs,lcl_derivs
            logical :: original_calc_status
            
        
        
            if (num_optimizable_params().ne.size(jacobian)) then
                call error("loss_feature_jacobian","Mismatch between length of Py and F95 jacobian")
            end if 
            
            !* need to read in net weights
            call parse_array_to_structure(flat_weights,net_weights)
            call copy_weights_to_nobiasT()

            !* allocate memory for feat. derivs wrt. basis fun. params
            call init_feature_array(gbl_derivs)

            original_calc_status = calc_feature_derivatives
            !* only use total energies in loss
            calc_feature_derivatives = .false.

            if (parallel) then
                write(*,*) 'parallel section not implemented yet'
                call exit(0)
            else

                do conf=1,data_sets(set_type)%nconf,1
                    call single_set(set_type,conf,gbl_derivs)
                end do !* end loop over confs
            end if

            calc_feature_derivatives = original_calc_status
            
            call parse_feature_format_to_array_jac(gbl_derivs,jacobian)
        end subroutine loss_feature_jacobian

        subroutine single_set(set_type,conf,lcl_feat_derivs)
            implicit none

            integer,intent(in) :: set_type,conf
            type(feature_info),intent(inout) :: lcl_feat_derivs

            !* scratch
            real(8) :: mxrcut,dr,zatm,zngh,tmpE,invN
            real(8),allocatable :: ultra_cart(:,:),ultra_z(:)
            integer,allocatable :: ultra_idx(:)
            integer :: atm,neigh,ftype,bond,ft
            logical :: calc_threebody
            type(feature_info) :: tmp_feat_derivs
           
            !* init param allocatables 
            call init_feature_array(tmp_feat_derivs)

            !* max cut off of all interactions
            mxrcut = maxrcut(0)

            !* whether three body interactions are present
            calc_threebody = threebody_features_present()

            !* get all nearest neighbours
            call get_ultracell(mxrcut,5000,set_type,conf,ultra_cart,ultra_idx,ultra_z)

            !* get atom-neighbour distances
            call calculate_twobody_info(set_type,conf,ultra_cart,ultra_z,ultra_idx)

            if (calc_threebody) then
                call calculate_threebody_info(set_type,conf,ultra_cart,ultra_z,ultra_idx)
            end if

            !* compute new features
            call calculate_all_features(set_type,conf)
           
            !* allocate mem. for dydx,a,z,a',delta
            call allocate_dydx(set_type,conf)
            call allocate_units(set_type,conf)
            
            !* forward prop for energy per atom
            call forward_propagate(set_type,conf)
    
            !* need dy_atm/dx for all atoms and features
            call backward_propagate(set_type,conf)
            
            
            !* (E_ref - \sum_i E_i)^2
            tmpE = sum(data_sets(set_type)%configs(conf)%current_ei) &
                    &-data_sets(set_type)%configs(conf)%ref_energy 

            invN = 1.0d0/dble(data_sets(set_type)%configs(conf)%n)

            if(loss_norm_type.eq.1) then
                tmpE = sign(1.0d0,tmpE) * invN
            else if (loss_norm_type.eq.2) then 
                tmpE = 2.0d0*sign(1.0d0,tmpE)*abs(tmpE) * (invN**2)
            end if
            tmpE = tmpE * loss_const_energy

            !* dy_atm / dparam = sum_ft dy_atm/dx_ft * dx_ft/dparam
            do atm=1,data_sets(set_type)%configs(conf)%n,1
                !* atomic number of central atom
                zatm = feature_isotropic(atm)%z_atom
               
                if (feature_isotropic(atm)%n.le.0) then
                    cycle
                end if
               
                !* dx/dparam for atm, too many neighbours to keep info for all
                !* atoms at once
                call zero_feature_info(tmp_feat_derivs)
                
                do neigh=1,feature_isotropic(atm)%n,1
                    !* atom-atom distance
                    dr = feature_isotropic(atm)%dr(neigh)

                    !* atomic number
                    zngh = feature_isotropic(atm)%z(neigh)

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
                                &tmp_feat_derivs%info(ft))
                    end do !* end loop over features
                end do !* end loop over two body neighbours to atm
              
                if (calc_threebody) then
                    if (feature_threebody_info(atm)%n.le.0) then
                        cycle
                    end if
                    
                    do bond=1,feature_threebody_info(atm)%n,1
                        do ft=1,feature_params%num_features,1
                            ftype = feature_params%info(ft)%ftype
                            
                            if (ftype.eq.featureID_StringToInt("atomic_number")) then
                                cycle
                            else if (feature_IsTwoBody(ftype)) then
                                cycle
                            end if
                            
                            call feature_ThreeBody_param_deriv(&
                                    &feature_threebody_info(atm)%dr(1:3,bond),&
                                    &feature_threebody_info(atm)%cos_ang(bond),&
                                    zatm,feature_threebody_info(atm)%z(1:2,bond),ft,&
                                    &tmp_feat_derivs%info(ft))
                        end do !* end loop over features
                    end do !* end loop over threebody bonds to atm
                end if !* end if three body interactions

                !* dy_atm / dparam = sum_ft dy_atm/dx_ft * dx_ft/dparam
                call append_atom_contribution(atm,tmp_feat_derivs,tmpE,lcl_feat_derivs)
            end do !* end loop over atoms

            deallocate(ultra_z)
            deallocate(ultra_idx)
            deallocate(ultra_cart)
            deallocate(feature_isotropic)
            if (calc_threebody) then
                deallocate(feature_threebody_info)
            end if


            !call add_feat_param_derivs(lcl_feat_derivs,tmp_feat_derivs,1.0d0)

        end subroutine single_set

        subroutine feature_TwoBody_param_deriv(dr,zatm,zngh,ft_idx,feature_deriv)
            implicit none

            real(8),intent(in) :: dr,zatm,zngh
            integer,intent(in) :: ft_idx
            type(feature_),intent(inout) :: feature_deriv

            !* scratch 
            integer :: ftype
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

                !lcl_feat_deriv%lambda = lcl_feat_deriv%lambda + (2.0d0**(1.0d0-xi))*&
                !        &*(1.0d0+lambda*cos_ang)**(xi-1.0d0) * xi*cos_ang*tmp_2*tmp_3

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
            integer :: ft,ftype

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
            end do
        end subroutine zero_feature_info

        subroutine add_feat_param_derivs(lcl_feat_derivs,tmp_feat_derivs,const)
            !* add tmp_feat_derivs*const to lcl_feat_derivs
            
            implicit none

            type(feature_info),intent(inout) :: lcl_feat_derivs
            type(feature_info),intent(in) :: tmp_feat_derivs
            real(8),intent(in) :: const

            !* scratch
            integer :: ft,ftype

            do ft=1,feature_params%num_features,1
                ftype = feature_params%info(ft)%ftype

                if (ftype.eq.featureID_StringToInt("acsf_behler-g2")) then
                    lcl_feat_derivs%info(ft)%eta = lcl_feat_derivs%info(ft)%eta + &
                            &tmp_feat_derivs%info(ft)%eta*const
                    
                    lcl_feat_derivs%info(ft)%xi = lcl_feat_derivs%info(ft)%xi + &
                            &tmp_feat_derivs%info(ft)%xi*const
                end if
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
                end if
            end do
            num_optimizable_params = cntr
        end function num_optimizable_params
end module feature_selection
