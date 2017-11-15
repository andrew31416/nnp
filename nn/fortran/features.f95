module features
    use config
    use io
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
            logical :: calc_threebody

            !* max cut off of all interactions
            mxrcut = maxrcut(0)
            
            !* whether threebody interactions are present
            calc_threebody = threebody_features_present()

            do set_type=1,2
                do conf=1,data_sets(set_type)%nconf
                    call get_ultracell(mxrcut,1000,set_type,conf,&
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
                    deallocate(feature_threebody_info)
                end do
            end do

        end subroutine calculate_features

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
                    if (ftype.eq.featureID_StringToInt("acsf_behler-g2")) then
                        call feature_behler_g2(atm,ii,ft_idx,data_sets(set_type)%configs(conf)%x(arr_idx,atm))

                        if (calc_feature_derivatives) then
                            call feature_behler_g2_deriv(atm,ii,ft_idx,&
                                &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%&
                                &vec(1:3,idx_to_contrib(ii)))
                        end if
                    else if (ftype.eq.featureID_StringToInt("acsf_normal-iso")) then
                        call feature_normal_iso(atm,ii,ft_idx,data_sets(set_type)%configs(conf)%x(arr_idx,atm))

                        if (calc_feature_derivatives) then
                            call feature_normal_iso_deriv(atm,ii,ft_idx,&
                                    &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%&
                                    &vec(1:3,idx_to_contrib(ii)))
                        end if
                    end if
                end if
            end do
            
            !* derivative wrt. central atm
            if (calc_feature_derivatives) then 
                if (ftype.eq.featureID_StringToInt("acsf_behler-g2")) then
                    call feature_behler_g2_deriv(atm,0,ft_idx,&
                            &data_sets(set_type)%configs(conf)%x_deriv(ft_idx,atm)%vec(1:3,1))
                else if (ftype.eq.featureID_StringToInt("acsf_normal-iso")) then
                    call feature_normal_iso_deriv(atm,0,ft_idx,&
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
                end if
            end do !* end loop ii over three body terms
            
            deallocate(bond_contributes)
        
        end subroutine feature_threebody
        
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
            real(8) :: tmp_feature,tap_ij,tap_jk,tap_ik
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

            tmp_feature = 2.0d0**(1.0d0-xi)*exp(-eta*(drij**2+drik**2+drjk**2)) * (1.0d0+lambda*cos_angle)**xi

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
                (   tap_ij*tap_ik*tap_jk*lambda*xi/(1.0d0+lambda*cos_angle)*dcosdrz +&
                &tap_ik*tap_jk*(tap_ij_deriv - 2.0d0*eta*tap_ij*drij)*drijdrz +&
                &tap_ij*tap_jk*(tap_ik_deriv - 2.0d0*eta*tap_ik*drik)*drikdrz +&
                &tap_ij*tap_ik*(tap_jk_deriv - 2.0d0*eta*tap_jk*drjk)*drjkdrz    )*tmp_feature*tmp_z
            end do
            
        end subroutine feature_behler_g4_deriv
        
        subroutine feature_normal_iso(atm,neigh_idx,ft_idx,current_val)
            implicit none

            integer,intent(in) :: atm,neigh_idx,ft_idx
            real(8),intent(inout) :: current_val

            !* scratch
            real(8) :: dr,tmp1,tmp2,tmp3,za,zb,rcut,fs,prec
            real(8) :: inv2pi,mean

            inv2pi = 0.15915494309d0

            !* atom-neigh_idx distance 
            dr  = feature_isotropic(atm)%dr(neigh_idx)
            
            !* symmetry function params
            za   = feature_params%info(ft_idx)%za
            zb   = feature_params%info(ft_idx)%zb
            fs   = feature_params%info(ft_idx)%fs
            prec = feature_params%info(ft_idx)%prec(1,1)
            mean = feature_params%info(ft_idx)%mean(1)
            rcut = feature_params%info(ft_idx)%rcut
            

            !* exponential
            tmp1 = sqrt(prec*inv2pi)*exp(-0.5d0*prec*(dr-mean)**2)

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
            real(8) :: fs,rcut,tmpz,prec,mean
            real(8) :: za,zb,inv2pi,prec_const
            integer :: ii,lim1,lim2
            
            inv2pi = 0.15915494309d0
            
            !* symmetry function params
            za   = feature_params%info(ft_idx)%za
            zb   = feature_params%info(ft_idx)%zb
            fs   = feature_params%info(ft_idx)%fs
            prec = feature_params%info(ft_idx)%prec(1,1)
            mean = feature_params%info(ft_idx)%mean(1)
            rcut = feature_params%info(ft_idx)%rcut

            prec_const = sqrt(inv2pi*prec)

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

                tmp1 =  prec_const*exp(-0.5d0*prec*(dr_scl-mean)**2)  *  (tap_deriv - &
                        &prec*(dr_scl-mean)*tap) 
                
                deriv_vec(:) = deriv_vec(:) + dr_vec(:)*tmp1*tmp2*tmpz
            end do
        end subroutine feature_normal_iso_deriv

end module
