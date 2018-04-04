module feature_selection
    use feature_util, only : get_ultracell, maxrcut, threebody_features_present
    use tapering, only : taper_1
    use io, only : error
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

            !* allocate memory for feat. derivs wrt. basis fun. params
            call init_feature_array(gbl_derivs)

            if (parallel) then
                write(*,*) 'parallel section not implemented yet'
                call exit(0)
            else

                do conf=1,data_sets(set_type)%nconf,1
                    call single_set(set_type,conf,gbl_derivs)
                end do !* end loop over confs
            end if
        end subroutine loss_feature_jacobian

        subroutine single_set(set_type,conf,lcl_feat_derivs)
            implicit none

            integer,intent(in) :: set_type,conf
            type(feature_info),intent(inout) :: lcl_feat_derivs

            !* scratch
            real(8) :: mxrcut,dr,zatm,zngh
            real(8),allocatable :: ultra_cart(:,:),ultra_z(:)
            integer,allocatable :: ultra_idx(:)
            integer :: atm,neigh,ftype,bond,ft
            logical :: calc_threebody

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

            do atm=1,data_sets(set_type)%configs(conf)%n,1
                !* atomic number of central atom
                zatm = feature_isotropic(atm)%z_atom
               
                if (feature_isotropic(atm)%n.le.0) then
                    cycle
                end if
                
                do neigh=1,feature_isotropic(atm)%n,1
                    !* atom-atom distance
                    dr = feature_isotropic(atm)%dr(neigh)

                    !* atomic number
                    zngh = feature_isotropic(atm)%z(neigh)

                    do ft=1,feature_params%num_features,1
                        ftype = feature_params%info(ft)%ftype

                        if (ftype.eq.featureID_StringToInt("atomic_number")) then
                            cycle
                        else if (feature_IsTwoBody(ftype).neqv..true.) then
                            cycle
                        end if

                        call feature_TwoBody_param_deriv(dr,zatm,zngh,ft,&
                                &lcl_feat_derivs%info(ft))
                    end do !* end loop over features
                end do !* end loop over two body neighbours to atm
              
                if (calc_threebody) then
                    if (feature_threebody_info(atm)%n.le.0) then
                        cycle
                    end if

                    do bond=1,feature_threebody_info(atm)%n,1
                        do ft=1,feature_params%num_features,1
                            if (ftype.eq.featureID_StringToInt("atomic_number")) then
                                cycle
                            else if (feature_IsTwoBody(ftype)) then
                                cycle
                            end if
                            
                            call feature_ThreeBody_param_deriv(&
                                    &feature_threebody_info(atm)%dr(1:3,bond),&
                                    &feature_threebody_info(atm)%cos_ang(bond),&
                                    zatm,feature_threebody_info(atm)%z(1:2,bond),ft,&
                                    &lcl_feat_derivs%info(ft))
                        end do !* end loop over features
                    end do !* end loop over threebody bonds to atm
                end if
            end do !* end loop over atoms

            deallocate(ultra_z)
            deallocate(ultra_idx)
            deallocate(ultra_cart)
            deallocate(feature_isotropic)
            if (calc_threebody) then
                deallocate(feature_threebody_info)
            end if

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

                feature_deriv%eta = feature_deriv%eta + 2.0d0*tmp_cnst*tmp2*&
                        &exp(prec*tmp2)
                
            else if (ftype.eq.featureID_StringToInt("acsf_normal-b2")) then
                !* params
                mu = feature_params%info(ft_idx)%mean(1)
                prec = feature_params%info(ft_idx)%prec(1,1)
           
                tmp1 = dr-mu
                tmp2 = -0.5d0*(tmp1**2)
            
                feature_deriv%mean(1) = feature_deriv%mean(1) + tmp_cnst*tmp1*prec*&
                        &exp(prec*tmp2)

                feature_deriv%prec(1,1) = feature_deriv%prec(1,1) + 2.0d0*tmp_cnst*tmp2*&
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
            real(8) :: zj,zk
            integer :: ftype

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

            !* all constants
            tmp_3 = scl_cnst*tap_ij*tap_ik*tap_jk*tmp_z
           
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

                lcl_feat_deriv%lambda = lcl_feat_deriv%lambda + (2.0d0**(1.0d0-xi))*&
                        &*(1.0d0+lambda*cos_ang)**(xi-1.0d0) * xi*cos_ang*tmp_2*tmp_3

                lcl_feat_deriv%eta = lcl_feat_deriv%eta - tmp_1*tmp_2*tmp_3*tmp_4
            
            else if (ftype.eq.featureID_StringToInt("acsf_behler-b3")) then
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
           
                lcl_feat_deriv%mean = lcl_feat_deriv%mean + lwork_1*exp_1 + lwork_2*exp_2
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

            do ft=1,feature_params%num_features,1
                ftype = feature_params%info(ft)%ftype

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
end module feature_selection
