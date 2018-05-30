program unittest

    use config
    use init
    use propagate
    use util
    use io
    use measures
    use features
    use feature_util
    use feature_selection
    use lookup

    implicit none

    call main()

    contains
        subroutine main()
            implicit none

            logical :: tests(1:13)
            
            !* net params
            integer :: num_nodes(1:2),nlf_type,fD

            !* features
            integer :: natm,nconf
            logical :: parallel,scale_features

            !* scratch
            integer :: ii
            character(len=24),dimension(13) :: test_string

            integer :: num_tests

            !* default status
            tests = .false.

            parallel = .false.
            !* do not scale features
            scale_features = .false.

            num_tests = size(tests)

            !* number of nodes
            num_nodes(1) = 5
            num_nodes(2) = 3

            !* nonlinear function
            nlf_type = 1
            
            !* features
            fD = 8
            natm = 5
            nconf = 2

            
            call unittest_header()
            
            !* generate atomic info
            call generate_testtrain_set(natm,nconf)

            !* initialise feature memory
            call generate_features(fD)

            !* calculate features and analytical derivatives
            call calculate_features(parallel,scale_features,.false.)

            !* generate lookup tables
            call init_lookup_tables()

            call initialise_net(num_nodes,nlf_type,fD)

            !* give biases non zero values
            !call random_number(net_weights%hl1(:,0))
            !call random_number(net_weights%hl2(:,0))
            !call random_number(net_weights%hl3(0))
            
            !----------------------!
            !* perform unit tests *!
            !----------------------!
            
            tests(1)  = test_dydw()                       ! dydw
            call test_loss_jac(tests(2:5))                ! d loss / dw
            tests(6)  = test_dydx()                       ! dydx
            tests(7)  = test_threebody_derivatives()      ! d cos(angle) /dr_i etc.
            tests(8)  = test_dxdr()                       ! d feature / d atom position
            tests(9)  = test_forces()                     ! - d E_tot / d r_atm 
            tests(10) = test_lookup_tables()              ! force comparison for lookup tables
            tests(11) = test_feature_selection_loss_jac() ! dloss / d(feature)param
            tests(12) = test_d2ydx2()                     ! d^2 y / dx^2 
            tests(13) = test_stress()                     ! stress tensor

            test_string(1)  = "dydw"
            test_string(2)  = "jac. loss (energy)"
            test_string(3)  = "jac. loss (forces)"
            test_string(4)  = "jac. loss (reg.)"
            test_string(5)  = "jac. loss (all)"
            test_string(6)  = "dydx"
            test_string(7)  = "dcos/dr"
            test_string(8)  = "dxdr"
            test_string(9)  = "forces"
            test_string(10) = "lookup table"
            test_string(11) = "jac. loss (features)"
            test_string(12) = "d^2y/dxdx"
            test_string(13) = "stress"
            
            do ii=1,num_tests
                call unittest_test(ii,test_string(ii),tests(ii))    
            end do

            call unittest_summary(tests)

            if (all(tests)) then
                !* success
                call exit(1)
            else
                !* failure
                call exit(0)
            end if
        end subroutine main

        subroutine generate_testtrain_set(natm,nconf)
            implicit none

            integer,intent(in) :: natm,nconf

            !* scratch 
            integer :: set_type,conf,ii
            real(8) :: tmpz

            do set_type=1,2
                data_sets(set_type)%nconf = nconf
                allocate(data_sets(set_type)%configs(nconf))

                do conf=1,nconf
                    allocate(data_sets(set_type)%configs(conf)%r(3,natm))
                    allocate(data_sets(set_type)%configs(conf)%z(natm))
                    allocate(data_sets(set_type)%configs(conf)%current_ei(natm))
                    allocate(data_sets(set_type)%configs(conf)%current_fi(3,natm))
                    allocate(data_sets(set_type)%configs(conf)%ref_fi(3,natm))
                    
                    data_sets(set_type)%configs(conf)%cell = 0.0d0
                    !data_sets(set_type)%configs(conf)%cell(1,1) = 4.0d0
                    data_sets(set_type)%configs(conf)%cell(1,1) = 6.5d0
                    data_sets(set_type)%configs(conf)%cell(2,1) = 0.1d0
                    data_sets(set_type)%configs(conf)%cell(3,1) = 0.2d0
                    data_sets(set_type)%configs(conf)%cell(1,2) = 0.1d0
                    !data_sets(set_type)%configs(conf)%cell(2,2) = 4.0d0
                    data_sets(set_type)%configs(conf)%cell(2,2) = 6.5d0
                    data_sets(set_type)%configs(conf)%cell(3,2) = 0.3d0
                    data_sets(set_type)%configs(conf)%cell(1,3) = -0.1d0
                    data_sets(set_type)%configs(conf)%cell(2,3) = 0.2d0
                    !data_sets(set_type)%configs(conf)%cell(3,3) = 4.0d0
                    data_sets(set_type)%configs(conf)%cell(3,3) = 6.5d0
                    data_sets(set_type)%configs(conf)%n = natm
                    call random_number(data_sets(set_type)%configs(conf)%ref_energy)
                    data_sets(set_type)%configs(conf)%ref_fi = 0.0d0

                    !data_sets(set_type)%configs(conf)%r(:,:) = 0.0d0
                    !data_sets(set_type)%configs(conf)%r(1,1) = 0.0d0
                    !data_sets(set_type)%configs(conf)%r(1,2) = 1.0d0
                    !data_sets(set_type)%configs(conf)%r(1,3) = 2.0d0
                

                    do ii=1,3
                        call random_number(data_sets(set_type)%configs(conf)%r(ii,:))
                        data_sets(set_type)%configs(conf)%r(ii,:) = &
                                &data_sets(set_type)%configs(conf)%r(ii,:)*&
                                &data_sets(set_type)%configs(conf)%cell(ii,ii)
                    end do
                    
                    do ii=1,natm
                        if (ii.lt.natm/2) then
                            tmpz = 1.0d0
                        else
                            tmpz = 6.0d0
                        end if
                        data_sets(set_type)%configs(conf)%z(ii) = tmpz
                    end do
                end do
            end do
        end subroutine generate_testtrain_set

        subroutine generate_features(fD)
            implicit none

            integer,intent(in) :: fD

            integer :: set_type,conf
            real(8) :: rcut

            allocate(feature_params%info(fD))
            feature_params%pca = .false.
            feature_params%pca_threshold = 0.0d0
            feature_params%num_features = fD


            rcut = 7.0d0
          
            !* test feature 1 
            feature_params%info(1)%ftype = featureID_StringToInt("atomic_number")    
            
            !* test feature 2
            feature_params%info(2)%ftype = featureID_StringToInt("acsf_behler-g1")            
            feature_params%info(2)%rcut = rcut
            feature_params%info(2)%fs = 0.2d0
            call random_number(feature_params%info(2)%za)
            call random_number(feature_params%info(2)%zb)
            
            !* test feature 3
            feature_params%info(3)%ftype = featureID_StringToInt("acsf_behler-g2")            
            feature_params%info(3)%rcut = rcut
            feature_params%info(3)%fs = 0.2d0
            call random_number(feature_params%info(3)%eta)
            call random_number(feature_params%info(3)%rs)
            call random_number(feature_params%info(3)%za)
            call random_number(feature_params%info(3)%zb)
            
            !* test feature 4
            feature_params%info(4)%ftype = featureID_StringToInt("acsf_normal-b2")
            feature_params%info(4)%rcut = rcut - 1.0d0
            feature_params%info(4)%fs = 0.2d0
            call random_number(feature_params%info(4)%za)
            call random_number(feature_params%info(4)%zb)
            allocate(feature_params%info(4)%prec(1,1))
            call random_number(feature_params%info(4)%prec(1,1)) 
            allocate(feature_params%info(4)%mean(1))
            call random_number(feature_params%info(4)%mean(1)) 
            feature_params%info(4)%sqrt_det = sqrt(feature_params%info(4)%prec(1,1))

            !* test feature 5
            feature_params%info(5)%ftype = featureID_StringToInt("acsf_behler-g4")
            feature_params%info(5)%rcut = 4.0d0
            feature_params%info(5)%fs = 0.2d0
            call random_number(feature_params%info(5)%lambda)
            call random_number(feature_params%info(5)%xi) 
            call random_number(feature_params%info(5)%eta) 
            call random_number(feature_params%info(5)%za)
            call random_number(feature_params%info(5)%zb)
            feature_params%info(5)%scl_cnst = 1.0d0

            !* test feature 6
            feature_params%info(6)%ftype = featureID_StringToInt("acsf_behler-g5")
            feature_params%info(6)%rcut = 3.1d0
            !feature_params%info(6)%rcut = 4.0d0
            feature_params%info(6)%fs = 0.2d0
            call random_number(feature_params%info(6)%lambda)
            call random_number(feature_params%info(6)%xi) 
            call random_number(feature_params%info(6)%eta) 
            call random_number(feature_params%info(6)%za)
            call random_number(feature_params%info(6)%zb)
            
            !* test feature 7
            feature_params%info(7)%ftype = featureID_StringToInt("acsf_normal-b3")
            feature_params%info(7)%rcut = 3.0d0
            feature_params%info(7)%fs = 0.3d0
            allocate(feature_params%info(7)%prec(3,3))
            allocate(feature_params%info(7)%mean(3))
            call random_number(feature_params%info(7)%prec)
            call random_number(feature_params%info(7)%za)
            call random_number(feature_params%info(7)%zb)
            feature_params%info(7)%mean(1) = 4.2d0
            feature_params%info(7)%mean(2) = 2.1d0
            feature_params%info(7)%mean(3) = 0.3d0
            feature_params%info(7)%sqrt_det = sqrt(matrix_determinant(feature_params%info(7)%prec))

            !* feature 8 : 2body fourier linear model
            feature_params%info(8)%ftype = featureID_StringToInt("acsf_fourier-b2")
            feature_params%info(8)%rcut = 5.6d0
            feature_params%info(8)%fs = 0.1d0
            allocate(feature_params%info(8)%linear_w(10))
            call random_number(feature_params%info(8)%linear_w)

            do set_type=1,2
                do conf=1,data_sets(set_type)%nconf
                    allocate(data_sets(set_type)%configs(conf)%x(fD+1,data_sets(set_type)%configs(conf)%n))
                    allocate(data_sets(set_type)%configs(conf)%x_deriv(fD,data_sets(set_type)%configs(conf)%n))
                end do
            end do
        end subroutine generate_features

        logical function test_dydw()
            implicit none

            logical :: test_result
    
            !* scratch
            integer :: ww,ii,jj,atm,conf,set_type
            real(8),allocatable :: original_weights(:)
            real(8),allocatable :: dydw_flat(:)
            real(8) :: dw,w0,dy,num_val
            logical :: deriv_ok,all_ok

            allocate(original_weights(nwght))
            allocate(dydw_flat(nwght))

            !* initial weights
            call parse_structure_to_array(net_weights,original_weights)
            
            conf = 1
            set_type = 1
            atm = 1
    
            if (allocated(dydx)) then
                deallocate(dydx)
            end if
            call allocate_dydx(set_type,conf)
            call allocate_units(set_type,conf)

            dy = 0.0d0

            test_result = .false.
                
            !--------------------------!
            !* analytical derivatives *!
            !--------------------------!

            !* forward prop
            call forward_propagate(set_type,conf)
            
            !* back prop
            call backward_propagate(set_type,conf)
            call calculate_dydw(set_type,conf,atm)
          
            !* parse dy/dw into 1d array
            call parse_structure_to_array(dydw,dydw_flat)

            all_ok = .true.

            do ii=1,nwght,1
                w0 = original_weights(ii)
                
                deriv_ok = .false.

                do ww=2,8,1
                
                    dw = 1.0d0/(10.0d0**ww)

                    do jj=1,2,1
                        if (jj.eq.1) then
                            original_weights(ii) = w0 + dw
                        else 
                            original_weights(ii) = w0 - dw
                        end if
                        
                        !* read in weights
                        call parse_array_to_structure(original_weights,net_weights)
                        call copy_weights_to_nobiasT()
                        
                        !* forward propagate
                        call forward_propagate(set_type,conf)

                        if (jj.eq.1) then
                            dy = data_sets(set_type)%configs(conf)%current_ei(atm)
                        else 
                            dy = dy - data_sets(set_type)%configs(conf)%current_ei(atm)
                      end if
                    end do !* end loop over +/- dw

                    num_val = dy/(2.0d0*dw)

                    if (scalar_equal(num_val,dydw_flat(ii),dble(1e-7),dble(1e-10),.false.)) then
                        deriv_ok = .true.
                    end if
               
                    !* return to initial value
                    original_weights(ii) = w0
                end do !* end loop over finite differences
               
                if (deriv_ok.neqv..true.) then
                    all_ok = .false.
                    write(*,*) ii,'of',nwght,'not OK'
                end if    
            end do !* end loop over weights

            deallocate(dydx)
            test_dydw = all_ok
        end function test_dydw

        subroutine test_loss_jac(test_result)
            !* test the jacobian of energy, force, reg. loss functions *!

            implicit none

            logical,intent(out) :: test_result(1:4)

            !* scratch
            integer :: ii,jj,kk,ww,conf,atm,set_type
            real(8) :: dw,w0,dloss,tmp,se(1:3)
            real(8),dimension(:),allocatable :: num_jac,anl_jac,original_weights
            logical :: deriv_ok,all_ok
            integer :: loss_norm_type

            allocate(num_jac(nwght))
            allocate(anl_jac(nwght))
            allocate(original_weights(nwght))
            
            !* initial weights
            call parse_structure_to_array(net_weights,original_weights)

            conf= 1
            atm = 1
            set_type = 1

            loss_norm_type = 2
            dloss = 0.0d0

            do ii=1,4,1
                if (ii.eq.1) then
                    loss_const_energy = 1.0d0
                    loss_const_forces = 0.0d0
                    loss_const_reglrn = 0.0d0
                else if (ii.eq.2) then
                    loss_const_energy = 0.0d0
                    loss_const_forces = 1.0d0
                    loss_const_reglrn = 0.0d0
                else if(ii.eq.3) then
                    loss_const_energy = 0.0d0
                    loss_const_forces = 0.0d0
                    loss_const_reglrn = 1.0d0
                else
                    loss_const_energy = 1.0d0
                    loss_const_forces = 0.6d0
                    loss_const_reglrn = 0.2d0
                end if
                
                !* set loss function parameters
                call init_loss(loss_const_energy,loss_const_forces,loss_const_reglrn,loss_norm_type)
                
                !-------------------------------!
                !* analytical jacobian of loss *!
                !-------------------------------!

                call loss_jacobian(original_weights,set_type,.false.,anl_jac)
                
                all_ok = .true.

                do jj=1,nwght,1
                    w0 = original_weights(jj)
                
                    deriv_ok = .false.

                    do ww=2,15,1
                        !* finite difference
                        dw = 1.0d0/(5.0d0**(ww))

                        do kk=1,2,1
                            if (kk.eq.1) then
                                original_weights(jj) = w0 + dw
                            else
                                original_weights(jj) = w0 - dw
                            end if
                            
                            tmp = loss(original_weights,set_type,.false.,se)

                            if (kk.eq.1) then
                                dloss = tmp
                            else
                                dloss = dloss - tmp
                            end if

                            original_weights(jj) = w0
                        end do !* end loop +/- dw

                        num_jac(jj) = dloss / (2.0d0 * dw)

                        if (scalar_equal(num_jac(jj),anl_jac(jj),dble(1e-10),&
                        &dble(1e-10),.false.)) then
                            deriv_ok = .true.
                        end if

                    end do !* end loop over +/- dw

                    if (deriv_ok.neqv..true.) then
                        !write(*,*) 'failed for ',jj,'of',nwght,num_jac(jj),anl_jac(jj)
                        all_ok = .false.
                    end if

                end do !* end loop over weights

                
                test_result(ii) = all_ok
            end do !* end loop over loss terms
        end subroutine test_loss_jac
        
        logical function test_feature_selection_loss_jac()
            !* test the jacobian of energy, force, reg. loss functions *!

            implicit none

            !* scratch
            integer :: ii,conf,atm,set_type,ft,ftype,cntr,num_weights
            real(8) :: dloss
            real(8),dimension(:),allocatable :: anl_jac,original_weights
            logical,allocatable :: anl_jac_ok(:) 
            integer :: loss_norm_type,num_params,num_attributes(0:8)
            logical :: scale_feats = .false.

            allocate(original_weights(nwght))
            
            !* initial weights
            call parse_structure_to_array(net_weights,original_weights)

            conf= 1
            atm = 1
            set_type = 1

            loss_norm_type = 2
            dloss = 0.0d0

            loss_const_energy = 0.5d0
            loss_const_forces = 0.0d0
            loss_const_reglrn = 0.0d0
            
            !* set loss function parameters
            call init_loss(loss_const_energy,loss_const_forces,loss_const_reglrn,loss_norm_type)
            
            if (.not.allocated(set_neigh_info)) then
                allocate(set_neigh_info(data_sets(set_type)%nconf))
            end if

            !* init array
            num_params = num_optimizable_params()
            allocate(anl_jac(num_params))
            call loss_feature_jacobian(original_weights,set_type,scale_feats,.false.,anl_jac)
            allocate(anl_jac_ok(num_params))

            !-------------------------------!
            !* analytical jacobian of loss *!
            !-------------------------------!

            call loss_feature_jacobian(original_weights,set_type,scale_feats,.false.,anl_jac)
            
            !* number of fourier weights
            num_weights = size(feature_params%info(8)%linear_w)

            !* number of optimizable attributes for each feature
            num_attributes(featureID_StringToInt("atomic_number"))   = 0
            num_attributes(featureID_StringToInt("acsf_behler-g1"))  = 0
            num_attributes(featureID_StringToInt("acsf_behler-g2"))  = 2
            num_attributes(featureID_StringToInt("acsf_behler-g4"))  = 2
            num_attributes(featureID_StringToInt("acsf_behler-g5"))  = 2
            num_attributes(featureID_StringToInt("acsf_normal-b2"))  = 2
            num_attributes(featureID_StringToInt("acsf_normal-b3"))  = 9
            num_attributes(featureID_StringToInt("acsf_fourier-b2")) = num_weights
            num_attributes(featureID_StringToInt("devel_iso"))       = 0

            cntr = 1
            do ft=1,feature_params%num_features,1
                ftype = feature_params%info(ft)%ftype
                
                do ii=1,num_attributes(ftype),1
                    !* compare analytical and numerical approx
                    anl_jac_ok(cntr) = feature_selection_subsidiary_1(original_weights,&
                            &set_type,ft,ii,anl_jac(cntr))
                    cntr = cntr + 1
                end do !* loop over optimizable attributes for given feature

            end do !* end loop over features
            
            test_feature_selection_loss_jac = all(anl_jac_ok)

            deallocate(set_neigh_info)
        end function test_feature_selection_loss_jac

        logical function feature_selection_subsidiary_1(original_weights,set_type,&
        &ft,attribute,anl_res)
            implicit none

            integer,intent(in) :: set_type,ft,attribute
            real(8),intent(in) :: anl_res,original_weights(:)

            integer :: ww,plusminus,num_steps,cntr,ftype 
            real(8) :: x0,se(1:3),newvalue,dw
            real(8),allocatable :: num_jac(:)
            logical :: match_found

            ftype = feature_params%info(ft)%ftype

            if (ftype.eq.FeatureID_StringToInt("acsf_behler-g2")) then
                if (attribute.eq.1) then
                    x0 = feature_params%info(ft)%eta
                else if (attribute.eq.2) then
                    x0 = feature_params%info(ft)%rs
                else
                    call unittest_error("feature_selection_subsidiary_1",&
                            &"Implementation error")
                end if
            else if ((ftype.eq.FeatureID_StringToInt("acsf_behler-g4")).or.&
                     (ftype.eq.FeatureID_StringToInt("acsf_behler-g5"))) then
                
                if (attribute.eq.1) then
                    x0 = feature_params%info(ft)%xi
                else if (attribute.eq.2) then
                    x0 = feature_params%info(ft)%eta
                else
                    call unittest_error("feature_selection_subsidiary_1",&
                            &"Implementation error")
                end if
            else if (ftype.eq.FeatureID_StringToInt("acsf_normal-b2")) then
                if (attribute.eq.1) then
                    x0 = feature_params%info(ft)%prec(1,1)
                else if (attribute.eq.2) then
                    x0 = feature_params%info(ft)%mean(1)
                else
                    call unittest_error("feature_selection_subsidiary_1",&
                            &"Implementation error")
                end if
            else if (ftype.eq.FeatureID_StringToInt("acsf_normal-b3")) then
                if (attribute.eq.1) then
                    x0 = feature_params%info(ft)%prec(1,1) 
                else if (attribute.eq.2) then
                    x0 = feature_params%info(ft)%prec(1,2) 
                else if (attribute.eq.3) then
                    x0 = feature_params%info(ft)%prec(1,3) 
                else if (attribute.eq.4) then
                    x0 = feature_params%info(ft)%prec(2,2) 
                else if (attribute.eq.5) then
                    x0 = feature_params%info(ft)%prec(2,3) 
                else if (attribute.eq.6) then
                    x0 = feature_params%info(ft)%prec(3,3) 
                else if (attribute.eq.7) then
                    x0 = feature_params%info(ft)%mean(1) 
                else if (attribute.eq.8) then
                    x0 = feature_params%info(ft)%mean(2) 
                else if (attribute.eq.9) then
                    x0 = feature_params%info(ft)%mean(3) 
                else
                    call unittest_error("feature_selection_subsidiary_1",&
                            &"Implementation error")
                end if
            else if (ftype.eq.FeatureID_StringToInt("acsf_fourier-b2")) then
                x0 = feature_params%info(ft)%linear_w(attribute) 
            else
                call unittest_error("feature_selection_subsidiary_1",&
                        &"attempting to optimize wrong feature")
            end if

            !* number of finite differences attempted
            num_steps = 25
            allocate(num_jac(num_steps))
            cntr = 1

            do ww=-5,-5+num_steps-1,1
                !* finite difference
                dw = 1.0d0/(2.0d0**(ww))

                do plusminus=1,2,1
                    if (plusminus.eq.1) then
                        newvalue = x0 + dw
                    else
                        newvalue = x0 - dw
                    end if
                    
                    !* set new feature value
                    call feature_selection_subsidiary_2(ft,attribute,newvalue)
               
                    !* need to calc. features for all confs
                    call calculate_features_singleset(set_type,.false.,.false.,&
                            &.true.,.false.,.true.)
                
                    if (plusminus.eq.1) then
                        num_jac(cntr) = loss(original_weights,set_type,.false.,se)
                    else
                        num_jac(cntr) = (num_jac(cntr) - &
                                &loss(original_weights,set_type,.false.,se)) / (2.0d0*dw)
                    end if
                end do !* loop over +/-

                cntr = cntr + 1
            end do !* loop over ww
            
            !* reset feature attribute to original value
            call feature_selection_subsidiary_2(ft,attribute,x0)

            match_found = .false.
            do ww=1,num_steps,1
                if (scalar_equal(num_jac(ww),anl_res,dble(1e-5),dble(1e-14),.false.)) then
                    match_found = .true.
                end if
            end do
            feature_selection_subsidiary_1 = match_found
        end function feature_selection_subsidiary_1

        subroutine feature_selection_subsidiary_2(ft,attribute,newvalue)
            implicit none

            !* args
            integer,intent(in) :: ft,attribute
            real(8),intent(in) :: newvalue

            !* scratch
            integer :: ftype

            ftype = feature_params%info(ft)%ftype

            if (ftype.eq.FeatureID_StringToInt("acsf_behler-g2")) then
                if (attribute.eq.1) then
                    feature_params%info(ft)%eta = newvalue
                else if (attribute.eq.2) then
                    feature_params%info(ft)%rs = newvalue
                else
                    call unittest_error("feature_selection_subsidiary_2",&
                            &"Implementation error")
                end if
            else if ((ftype.eq.FeatureID_StringToInt("acsf_behler-g4")).or.&
                     (ftype.eq.FeatureID_StringToInt("acsf_behler-g5"))) then
                
                if (attribute.eq.1) then
                    feature_params%info(ft)%xi = newvalue
                else if (attribute.eq.2) then
                    feature_params%info(ft)%eta = newvalue
                else
                    call unittest_error("feature_selection_subsidiary_2",&
                            &"Implementation error")
                end if
            else if (ftype.eq.FeatureID_StringToInt("acsf_normal-b2")) then
                if (attribute.eq.1) then
                    feature_params%info(ft)%prec(1,1)= newvalue
                else if (attribute.eq.2) then
                    feature_params%info(ft)%mean(1)= newvalue
                else
                    call unittest_error("feature_selection_subsidiary_2",&
                            &"Implementation error")
                end if
            else if (ftype.eq.FeatureID_StringToInt("acsf_normal-b3")) then
                !* remember that matrix is symmetric
                if (attribute.eq.1) then
                    feature_params%info(ft)%prec(1,1) = newvalue
                else if (attribute.eq.2) then
                    feature_params%info(ft)%prec(1,2) = newvalue
                    feature_params%info(ft)%prec(2,1) = newvalue
                else if (attribute.eq.3) then
                    feature_params%info(ft)%prec(1,3) = newvalue
                    feature_params%info(ft)%prec(3,1) = newvalue
                else if (attribute.eq.4) then
                    feature_params%info(ft)%prec(2,2) = newvalue
                else if (attribute.eq.5) then
                    feature_params%info(ft)%prec(2,3) = newvalue
                    feature_params%info(ft)%prec(3,2) = newvalue
                else if (attribute.eq.6) then
                    feature_params%info(ft)%prec(3,3) = newvalue
                else if (attribute.eq.7) then
                    feature_params%info(ft)%mean(1) = newvalue
                else if (attribute.eq.8) then
                    feature_params%info(ft)%mean(2) = newvalue
                else if (attribute.eq.9) then
                    feature_params%info(ft)%mean(3) = newvalue
                else
                    call unittest_error("feature_selection_subsidiary_2",&
                            &"Implementation error")
                end if
            else if (ftype.eq.FeatureID_StringToInt("acsf_fourier-b2")) then
                feature_params%info(ft)%linear_w(attribute) = newvalue
            else
                call unittest_error("feature_selection_subsidiary_2",&
                        &"attempting to optimize wrong feature")
            end if

        end subroutine feature_selection_subsidiary_2

        logical function test_dydx()
            implicit none

            !* scratch 
            integer :: set_type,conf,atm,xx,ii,ww
            real(8) :: dw,x0
            real(8),allocatable :: num_dydx(:)
            logical,allocatable :: log_atms(:),cnf_atms(:)
            logical :: set_atms(1:2)
            logical :: deriv_ok

            do set_type=1,2
                allocate(cnf_atms(data_sets(set_type)%nconf))

                do conf=1,data_sets(set_type)%nconf
                    allocate(log_atms(data_sets(set_type)%configs(conf)%n))
                    allocate(num_dydx(D))
                  
                    if (allocated(dydx)) then
                        deallocate(dydx)
                    end if 
                    call allocate_dydx(set_type,conf)
                    call allocate_units(set_type,conf)
                    
                    do atm=1,data_sets(set_type)%configs(conf)%n
                        log_atms(atm) = .true.

                        !---------------------------!
                        !* analytical differential *!
                        !---------------------------!
                        
                        call forward_propagate(set_type,conf)
                        call backward_propagate(set_type,conf)

                        !--------------------------!
                        !* numerical differential *!
                        !--------------------------!
                        
                        do xx=1,D
                            x0 = data_sets(set_type)%configs(conf)%x(xx+1,atm)

                            deriv_ok = .false.

                            do ww=-2,3 
                                !* finite difference for feature
                                dw = 1.0d0/(10**ww)

                                do ii=1,2
                                    if (ii.eq.1) then
                                        data_sets(set_type)%configs(conf)%x(xx+1,atm) = x0 + dw
                                    else
                                        data_sets(set_type)%configs(conf)%x(xx+1,atm) = x0 - dw
                                    end if
                                    
                                    call forward_propagate(set_type,conf)
                                    
                                    if (ii.eq.1) then
                                        num_dydx(xx) = data_sets(set_type)%configs(conf)%current_ei(atm)
                                    else
                                        num_dydx(xx) = num_dydx(xx) - &
                                                &data_sets(set_type)%configs(conf)%current_ei(atm)
                                    end if
                                    
                                    data_sets(set_type)%configs(conf)%x(xx+1,atm) = x0 
                                end do !* end loop +/- dw
                                
                                num_dydx(xx) = num_dydx(xx) / (2.0d0*dw)

                                if (scalar_equal(num_dydx(xx),dydx(xx,atm),dble(1e-7),&
                                &dble(1e-10),.false.)) then
                                    deriv_ok = .true.
                                end if
                            end do !* end loop finite differences

                            if (deriv_ok.neqv..true.) then
                                log_atms(atm) = .false.
                            end if
                        end do !* end loop features
                        
                    end do !* end loop atoms
                
                    cnf_atms(conf) = all(log_atms)

                    deallocate(log_atms)
                    deallocate(num_dydx)
                    deallocate(dydx)
                end do !* end loop configurations

                set_atms(set_type) = all(cnf_atms)

                deallocate(cnf_atms)
            end do !* end loop data sets

            test_dydx = all(set_atms)
        end function test_dydx

        logical function test_dxdr()
            implicit none

            !* scratch
            integer :: ii,jj,kk,ll,dd,ww,set_type,conf,atm
            real(8) :: dw,x0
            real(8),allocatable :: num_dxdr(:,:)
            type(feature_derivatives),allocatable :: anl_deriv(:,:)
            logical :: deriv_matches,atom_ok,all_ok,set_arr(1:2)
            logical,allocatable :: conf_arr(:),atm_arr(:)
            logical :: atom_passes,parallel,scale_features,updating_features
            

            parallel = .false.
            scale_features = .false.
            updating_features = .false.

            do set_type=1,2
                allocate(conf_arr(data_sets(set_type)%nconf))

                do conf=1,data_sets(set_type)%nconf
                    !* finite difference of features for all atoms
                    allocate(num_dxdr(D,data_sets(set_type)%configs(conf)%n))
                    allocate(anl_deriv(D,data_sets(set_type)%configs(conf)%n))
                    
                    allocate(atm_arr(data_sets(set_type)%configs(conf)%n))
                    atm_arr(:) = .true.

                    !* deallocate previous mem
                    call deallocate_feature_deriv_info()

                    !* make sure feature derivative calculation is turned on
                    call switch_property("forces","on")
                    
                    !* calculate analytical derivatives
                    call calculate_features(scale_features,parallel,updating_features)
                    
                    !* copy numerical derivatives
                    do ii=1,data_sets(set_type)%configs(conf)%n
                        do jj=1,D
                            anl_deriv(jj,ii)%n = data_sets(set_type)%configs(conf)%x_deriv(jj,ii)%n
                            
                            if (anl_deriv(jj,ii)%n.ne.0) then
                                allocate(anl_deriv(jj,ii)%idx(anl_deriv(jj,ii)%n))
                                allocate(anl_deriv(jj,ii)%vec(3,anl_deriv(jj,ii)%n))
                                anl_deriv(jj,ii)%idx(:) = &
                                        &data_sets(set_type)%configs(conf)%x_deriv(jj,ii)%idx(:)
                                anl_deriv(jj,ii)%vec(:,:) = &
                                        &data_sets(set_type)%configs(conf)%x_deriv(jj,ii)%vec(:,:)
                            end if
                        end do
                    end do
                    
                    !* feat. derivs. are expensive for numerical differential
                    call switch_property("forces","off")

                    do atm=1,data_sets(set_type)%configs(conf)%n
                        do dd=1,3,1
                            !* real space coordinate
                            x0 = data_sets(set_type)%configs(conf)%r(dd,atm)
                           
                            !* if one of finite difference is OK, atom_passes = True
                            atom_passes = .false. 
                            do ww=3,8,1
        
                                !-----------------------------!
                                !* numerical differentiation *!
                                !-----------------------------!

                                !* finite difference (A)
                                dw = 1.0d0/(10.0d0**ww)
                                
                                do ii=1,2,1
                                    if (ii.eq.1) then
                                        data_sets(set_type)%configs(conf)%r(dd,atm) = x0 + dw
                                    else
                                        data_sets(set_type)%configs(conf)%r(dd,atm) = x0 - dw
                                    end if
                                    
                                    !* deallocate previous mem
                                    call deallocate_feature_deriv_info()
                                    
                                    !* calculate features
                                    call calculate_features(scale_features,parallel,&
                                            &updating_features)
                                    
                                    if (ii.eq.1) then
                                        num_dxdr(:,:) = data_sets(set_type)%configs(conf)%x(2:,:)
                                    else
                                        num_dxdr(:,:) = ( num_dxdr(:,:) - &
                                                &data_sets(set_type)%configs(conf)%x(2:,:) ) / (2.0d0*dw)
                                    end if
                                    
                                    !* return coordinate to original value
                                    data_sets(set_type)%configs(conf)%r(dd,atm) = x0
                                end do !* end loop +/- dw
                                
                                !---------------------------------------!
                                !* analytical and numerical comparison *!
                                !---------------------------------------!

                                all_ok = .true.
                                
                                !* search for all terms with atm in
                                do jj=1,data_sets(set_type)%configs(conf)%n,1    
                                    atom_ok = .true.
                                    do kk=1,D,1 
                                        deriv_matches = .true.
                                        if (scalar_equal(num_dxdr(kk,jj),0.0d0,dble(1e-10),&
                                        &dble(1e-10),.false.).neqv..true.) then
                                            deriv_matches = .false.
                                            do ll=1,anl_deriv(kk,jj)%n
                                                !* loop over atoms which contribute to feature kk,atom jj
                                                if (anl_deriv(kk,jj)%idx(ll).eq.atm) then 
                                                    if ( scalar_equal(num_dxdr(kk,jj),&
                                                    &anl_deriv(kk,jj)%vec(dd,ll),dble(1e-7),&
                                                    &dble(1e-7),.false.) ) then
                                                        deriv_matches = .true.    
                                                    end if
                                                end if  
                                            end do !* end loop over contributing atoms to (kk,jj)
                                        end if
                                        
                                        if (deriv_matches.neqv..true.) then
                                            !* one of feature derivs for this atom is wrong
                                            atom_ok = .false.
                                        end if
                                    end do !* end loop over features


                                    if (atom_ok.neqv..true.) then
                                        all_ok = .false.
                                    end if
                                end do !* end loop over atoms jj
                

                                if (all_ok) then
                                    atom_passes = .true.
                                end if

                                !if (all_ok.neqv..true.) then
                                !    atm_arr(atm) = .false.
                                !end if
                            end do !* end loop over finite difference

                            if (atom_passes.neqv..true.) then
                                !* none of the finite differences gave a correct solution
                                atm_arr(atm) = .false.
                            end if
                        end do !* end loop over dimensions
                    end do !* end loop over atoms in structure

                    conf_arr(conf) = all(atm_arr)
                    
                    deallocate(atm_arr)
                    deallocate(num_dxdr)
                    deallocate(anl_deriv)
                end do  !* end loop over structures in set
                set_arr(set_type) = all(conf_arr)
                
                deallocate(conf_arr)
            end do !* end loop over test/train sets
            
            test_dxdr = all(set_arr)
        end function test_dxdr

        logical function test_threebody_derivatives()
            implicit none

            !* scratch
            integer :: set_type,conf,atm,dd,ii,jj,kk,ww,bond
            integer :: nmax
            real(8) :: x0,dw,ftol,rtol
            real(8),allocatable :: ultraz(:),ultracart(:,:)
            type(feature_info_threebody),allocatable :: threebody_ref(:)
            type(feature_info_threebody),allocatable :: threebody_dif(:)
            integer,allocatable :: ultraidx(:),deriv_ok(:,:,:),deriv_ok_cos(:,:)
            logical :: bad_deriv,dw_ok,all_ok

            if (threebody_features_present().neqv..true.) then
                test_threebody_derivatives = .true.
            end if

            all_ok = .true.

            ftol = dble(1e-7)
            rtol = dble(1e-8)


            !do set_type=1,2
            do set_type=1,1
                !* initialise set_neigh_info
                call init_set_neigh_info(set_type)
                
                !do conf=1,data_sets(set_type)%nconf
                do conf=1,1
                    !* all interacting atom projections
                    call get_ultracell(maxrcut(0),5000,set_type,conf,&
                            &ultracart,ultraidx,ultraz)
                
                    call calculate_threebody_info(set_type,conf,ultracart,ultraz,ultraidx)
                    deallocate(ultracart)
                    deallocate(ultraz)
                    deallocate(ultraidx)
                    
                    nmax = set_neigh_info(conf)%threebody(1)%n
                    do atm=2,data_sets(set_type)%configs(conf)%n
                        if (set_neigh_info(conf)%threebody(atm)%n.gt.nmax) then
                            nmax = set_neigh_info(conf)%threebody(atm)%n
                        end if
                    end do

                    allocate(deriv_ok(3,nmax,data_sets(set_type)%configs(conf)%n))
                    allocate(deriv_ok_cos(nmax,data_sets(set_type)%configs(conf)%n))
                    deriv_ok = 0
                    deriv_ok_cos = 0

                    !* copy threebody info
                    call copy_threebody_feature_info(set_neigh_info(conf)%threebody,threebody_ref)
                    deallocate(set_neigh_info(conf)%threebody)
                    !deallocate(feature_threebody_info) ! DEP.

                    do atm=1,data_sets(set_type)%configs(conf)%n
                        do dd=1,3,1
                            x0 = data_sets(set_type)%configs(conf)%r(dd,atm)

                            dw_ok = .false.

                            do ww=4,7,1
                                dw = 1.0d0/(10.0d0**dble(ww))
                            
                                do ii=1,2
                                    if (ii.eq.1) then
                                        data_sets(set_type)%configs(conf)%r(dd,atm) = x0 + dw
                                    else
                                        data_sets(set_type)%configs(conf)%r(dd,atm) = x0 - dw
                                    end if

                                    call get_ultracell(maxrcut(0),5000,set_type,conf,&
                                            &ultracart,ultraidx,ultraz)
                                    
                                    call calculate_threebody_info(set_type,conf,ultracart,&
                                            &ultraz,ultraidx)
                                    deallocate(ultracart)
                                    deallocate(ultraz)
                                    deallocate(ultraidx)
                                    
                                    if (ii.eq.1) then
                                        call copy_threebody_feature_info(&
                                                &set_neigh_info(conf)%threebody,threebody_dif) 
                                    else
                                        !* compute finite difference
                                        do jj=1,data_sets(set_type)%configs(conf)%n
                                            if (size(set_neigh_info(conf)%threebody(jj)%cos_ang)&
                                            &.ne.size(threebody_dif(jj)%cos_ang)) then
                                                write(*,*) 'dw = ',dw,&
                                                        &'too large in threebody deriv test'
                                                call exit(0)
                                            end if
                                            
                                            threebody_dif(jj)%cos_ang = (threebody_dif(jj)%cos_ang &
                                                    &- set_neigh_info(conf)%threebody(jj)%cos_ang)&
                                                    &/(2.0d0*dw)
                                            threebody_dif(jj)%dr = (threebody_dif(jj)%dr - &
                                                    &set_neigh_info(conf)%threebody(jj)%dr)/&
                                                    &(2.0d0*dw)
                                        end do
                                    end if

                                    deallocate(set_neigh_info(conf)%threebody)
                                    !deallocate(feature_threebody_info)
                                end do !* end loop over +/- dw

                                !* if true, error in derivatives
                                bad_deriv = .false.
                                
                                !* compare analytical to numeric differential
                                do jj=1,data_sets(set_type)%configs(conf)%n
                                    do bond=1,threebody_dif(jj)%n
                                        do kk=1,3
                                            !-----------------------!
                                            !* check dr derivative *!
                                            !-----------------------!

                                            if (scalar_equal(threebody_dif(jj)%dr(kk,bond),0.0d0,&
                                            &dble(1e-10),dble(1e-10),.false.).neqv..true.) then
                                                if (atm.eq.jj) then
                                                    if (kk.le.2) then
                                                        if (scalar_equal(threebody_dif(jj)%dr(kk,bond),&
                                                        &-threebody_ref(jj)%drdri(dd,(kk-1)*2+1,bond),&
                                                        &ftol,rtol,.false.).eqv..true.) then
                                                            deriv_ok(kk,bond,jj) = 1
                                                        else
                                                            if (deriv_ok(kk,bond,jj).eq.0) then
                                                                deriv_ok(kk,bond,jj) = -1
                                                            end if
                                                        end if
                                                    else
                                                        ! djk/dri_d
                                                        if (scalar_equal(threebody_dif(jj)%dr(kk,bond),&
                                                        &threebody_ref(jj)%drdri(dd,6,bond),ftol,&
                                                        &rtol,.false.).eqv..true.) then
                                                            deriv_ok(kk,bond,jj) = 1
                                                        else
                                                            if (deriv_ok(kk,bond,jj).eq.0) then
                                                                deriv_ok(kk,bond,jj) = -1
                                                            end if
                                                        end if
                                                    end if
                                                end if !* end if atm == ii
                                                if (atm.eq.threebody_dif(jj)%idx(1,bond)) then
                                                    if (kk.eq.1) then
                                                        if (scalar_equal(threebody_dif(jj)%dr(kk,bond),&
                                                        &threebody_ref(jj)%drdri(dd,1,bond),ftol,&
                                                        &rtol,.false.).eqv..true.) then
                                                            deriv_ok(kk,bond,jj) = 1
                                                        else
                                                            if (deriv_ok(kk,bond,jj).eq.0) then
                                                                deriv_ok(kk,bond,jj) = -1
                                                            end if
                                                        end if
                                                    else if (kk.eq.2) then
                                                        if (scalar_equal(threebody_dif(jj)%dr(kk,bond),&
                                                        &threebody_ref(jj)%drdri(dd,4,bond),ftol,&
                                                        &rtol,.false.).eqv..true.) then
                                                            deriv_ok(kk,bond,jj) = 1
                                                        else
                                                            if (deriv_ok(kk,bond,jj).eq.0) then
                                                                deriv_ok(kk,bond,jj) = -1
                                                            end if
                                                        end if
                                                    else if (kk.eq.3) then
                                                        if (scalar_equal(threebody_dif(jj)%dr(kk,bond),&
                                                        &-threebody_ref(jj)%drdri(dd,5,bond),ftol,&
                                                        &rtol,.false.).eqv..true.) then
                                                            deriv_ok(kk,bond,jj) = 1
                                                        else
                                                            if (deriv_ok(kk,bond,jj).eq.0) then
                                                                deriv_ok(kk,bond,jj) = -1
                                                            end if
                                                        end if
                                                    end if
                                                end if !* end if atom == jj
                                                if (atm.eq.threebody_dif(jj)%idx(2,bond)) then
                                                    if (kk.eq.1) then
                                                        if (scalar_equal(threebody_dif(jj)%dr(kk,bond),&
                                                        &threebody_ref(jj)%drdri(dd,2,bond),ftol,&
                                                        &rtol,.false.).eqv..true.) then
                                                            deriv_ok(kk,bond,jj) = 1
                                                        else
                                                            if (deriv_ok(kk,bond,jj).eq.0) then
                                                                deriv_ok(kk,bond,jj) = -1
                                                            end if
                                                        end if
                                                    else if (kk.eq.2) then
                                                        if (scalar_equal(threebody_dif(jj)%dr(kk,bond),&
                                                        &threebody_ref(jj)%drdri(dd,3,bond),ftol,&
                                                        &rtol,.false.).eqv..true.) then
                                                            deriv_ok(kk,bond,jj) = 1
                                                        else
                                                            if (deriv_ok(kk,bond,jj).eq.0) then
                                                                deriv_ok(kk,bond,jj) = -1
                                                            end if
                                                        end if
                                                    else if (kk.eq.3) then
                                                        if (scalar_equal(threebody_dif(jj)%dr(kk,bond),&
                                                        &threebody_ref(jj)%drdri(dd,5,bond),ftol,&
                                                        &rtol,.false.).eqv..true.) then
                                                            deriv_ok(kk,bond,jj) = 1
                                                        else
                                                            if (deriv_ok(kk,bond,jj).eq.0) then
                                                                deriv_ok(kk,bond,jj) = -1
                                                            end if
                                                        end if
                                                    end if
                                                end if !* end if atom == kk
                                                
                                            end if  !* end if derivative type = dr

                                        end do
                                        
                                        !------------------------!
                                        !* cos_{ijk} derivative *!
                                        !------------------------!

                                        if (scalar_equal(threebody_dif(jj)%cos_ang(bond),0.0d0,&
                                        &dble(1e-10),dble(1e-15),.false.).neqv..true.) then
                                            if (atm.eq.jj) then
                                                if (scalar_equal(threebody_dif(jj)%cos_ang(bond),&
                                                &threebody_ref(jj)%dcos_dr(dd,3,bond),ftol,&
                                                &rtol,.false.)) then
                                                    deriv_ok_cos(bond,jj) = 1
                                                else
                                                    if (deriv_ok_cos(bond,jj).eq.0) then
                                                        deriv_ok_cos(bond,jj) = -1
                                                    end if
                                                end if
                                            end if 
                                        end if
                                    end do !* end loop over bonds
                                end do !* end loop over jj

                                if (bad_deriv.neqv..true.) then
                                    dw_ok = .true.
                                end if


                                deallocate(threebody_dif)
                            end do !* end loop over finite difference magnitude
                            
                            do jj=1,data_sets(set_type)%configs(conf)%n
                                do bond=1,threebody_ref(jj)%n
                                    do kk=1,3
                                        if (deriv_ok(kk,bond,jj).eq.-1) then
                                            !write(*,*) 'drdri fail:',kk,bond,jj,atm
                                            all_ok = .false.
                                        end if
                                    end do

                                    if (deriv_ok_cos(bond,jj).eq.-1) then
                                        all_ok = .false.
                                        write(*,*) 'dcosdr fail:',jj,bond
                                    end if
                                end do
                            end do 
                            
                        
                        end do !* end loop over dimenions
                    end do !* end loop over local cell atoms
                    
                    deallocate(deriv_ok)
                    deallocate(deriv_ok_cos)
                    deallocate(threebody_ref)
                end do !* end loop over configs
            
                deallocate(set_neigh_info)
            end do !* end loop over sets

            test_threebody_derivatives = all_ok
        end function test_threebody_derivatives

        logical function test_forces()
            implicit none

            !* scratch
            integer :: set_type,conf,atm,dd,ww,ii
            real(8) :: dw,num_val,etot,x0
            real(8),allocatable :: anl_forces(:,:)
            logical,allocatable :: atms_ok(:),conf_ok(:)
            logical :: dd_ok,set_ok(1:2),parallel,scale_features,updating_features

            parallel = .false.
            scale_features = .false.
            updating_features = .false.

            do set_type=1,2,1
                allocate(conf_ok(data_sets(set_type)%nconf))

                do conf=1,data_sets(set_type)%nconf,1
                    if (allocated(dydx)) then
                        deallocate(dydx)
                    end if
                    call allocate_dydx(set_type,conf)
                    call allocate_units(set_type,conf)
                    
                    !* make sure we'are calculating derivatives
                    call switch_property("forces","on")

                    call deallocate_feature_deriv_info()
                    call calculate_features(scale_features,parallel,updating_features)

                    allocate(anl_forces(3,data_sets(set_type)%configs(conf)%n))
                    
                    do atm=1,data_sets(set_type)%configs(conf)%n,1
                        call forward_propagate(set_type,conf)
                        call backward_propagate(set_type,conf)
                    end do
                    
                    !* analytical forces
                    call calculate_forces(set_type,conf)
                    anl_forces(:,:) = data_sets(set_type)%configs(conf)%current_fi(:,:)
                    
                    !* don't need feature derivatives for numerical force
                    call switch_property("forces","off")
                    
                    
                    allocate(atms_ok(data_sets(set_type)%configs(conf)%n))
                    atms_ok(:) = .true.

                    do atm=1,data_sets(set_type)%configs(conf)%n,1
                        do dd=1,3,1
                            x0 = data_sets(set_type)%configs(conf)%r(dd,atm)
                            dd_ok = .false.
                            do ww=1,10,1
                                !* finite difference
                                dw = dble(1.0d0/(5.0d0**ww))
                                
                                do ii=1,2,1
                                    if (ii.eq.1) then
                                        data_sets(set_type)%configs(conf)%r(dd,atm) = x0 + dw
                                    else
                                        data_sets(set_type)%configs(conf)%r(dd,atm) = x0 - dw
                                    end if
                                    
                                    call deallocate_feature_deriv_info()
                                    call calculate_features(scale_features,parallel,&
                                            &updating_features)

                                    call forward_propagate(set_type,conf)

                                    !* total energy
                                    etot = sum(data_sets(set_type)%configs(conf)%current_ei)
                                    
                                    if (ii.eq.1) then
                                        num_val = etot
                                    else
                                        !* - d E_tot / d r_atm,dd
                                        num_val = -(num_val - etot)
                                    end if
                                end do !* end loop over +/- dw
                                
                                num_val = num_val/(2.d0*dw)


                                !* numerical vs. analytical
                                if (scalar_equal(num_val,anl_forces(dd,atm),dble(1e-5),&
                                &dble(1e-10),.false.)) then
                                    dd_ok = .true.
                                end if
                                
                                !* book keeping
                                data_sets(set_type)%configs(conf)%r(dd,atm) = x0 

                            end do !* end loop over finite difference magnitude

                            if (dd_ok.neqv..true.) then
                                !* derivative fails atom
                                atms_ok(atm) = .false.
                            end if
                        end do !* end loop over dimensions dd
                    end do !* end loop over local cell atoms atm

                    conf_ok(conf) = all(atms_ok)

                    deallocate(dydx)
                    deallocate(anl_forces)
                    deallocate(atms_ok)
                end do !* end loop over confs in set

                set_ok(set_type) = all(conf_ok)

                deallocate(conf_ok)
            end do !* end loop over data sets

            test_forces = all(set_ok)
        end function test_forces

        logical function test_d2ydx2()
            implicit none

            !* scratch
            integer :: set_type,conf,atm,kk,ww,ii
            real(8) :: x0,xnew,net_output(1:3),num_d2ydx2,dx
            real(8),allocatable :: anl_d2ydx2(:,:)
            logical,allocatable :: conf_ok(:,:)
            logical :: all_confs_ok = .true.
            
            do set_type=1,1
                do conf=1,1
                    call forward_propagate(set_type,conf)
                    call backward_propagate(set_type,conf)

                    !* compute full hessian
                    call calculate_d2ydxdx(set_type,conf)
                    
                    if (allocated(anl_d2ydx2)) then
                        deallocate(anl_d2ydx2)
                    end if
                    allocate(anl_d2ydx2(D,data_sets(set_type)%configs(conf)%n))
                    if (allocated(conf_ok)) then
                        deallocate(conf_ok)
                    end if
                    allocate(conf_ok(D,data_sets(set_type)%configs(conf)%n))
                    conf_ok = .false.

                    do atm=1,data_sets(set_type)%configs(conf)%n,1
                        do kk=1,feature_params%num_features,1
                            !* d^2 y / dx^2 are diagonal elements of hessian
                            anl_d2ydx2(kk,atm) = d2ydxdx(kk,kk,atm)
                        end do
                    end do
                    
                    do atm=1,data_sets(set_type)%configs(conf)%n,1
                        do kk=1,D
                            !* coordinate value
                            x0 = data_sets(set_type)%configs(conf)%x(kk+1,atm)

                            do ww=-5,1
                                !* loop over a number of finite differences
                                dx = dble(1.0d0/(5.0d0**ww))

                                do ii=1,3,1
                                    if (ii.eq.1) then
                                        xnew = x0 - dx
                                    else if (ii.eq.2) then
                                        xnew = x0
                                    else if (ii.eq.3) then
                                        xnew = x0 + dx
                                    end if

                                    data_sets(set_type)%configs(conf)%x(kk+1,atm) = xnew

                                    !* don't need to compute forces fortunately
                                    call forward_propagate(set_type,conf)

                                    net_output(ii) = &
                                            &data_sets(set_type)%configs(conf)%current_ei(atm)
                                end do !* end loop over -dx,0,+dx

                                num_d2ydx2 = (net_output(3) - 2.0d0*net_output(2) + &
                                        &net_output(1))/(dx**2)
                                
                                if (scalar_equal(num_d2ydx2,anl_d2ydx2(kk,atm),&
                                &dble(1e-7),dble(1e-25),.false.)) then
                                    !* correct answer
                                    conf_ok(kk,atm) = .true.
                                end if
                            end do !* end loop over ww
                            
                            !* restore original coordinate value
                            data_sets(set_type)%configs(conf)%x(kk+1,atm) = x0
                        end do !* end loop over feature coordinates
                    end do !* end loop over atoms

                    if (.not.all(conf_ok)) then
                        all_confs_ok = .false.
                    end if
                end do !* end loop over confs
            end do !* loop over sets
    
            test_d2ydx2 = all_confs_ok
        end function test_d2ydx2

        logical function test_stress()
            implicit none

            integer :: set_type,conf,indices(1:2,1:3)
            real(8) :: stress(1:3,1:3),val1,val2
            integer :: ii
            logical :: ok = .true.

            !* diagonal index pairs to compare - stress should be symmetric
            indices(1,1) = 1
            indices(2,1) = 2
            indices(1,2) = 1
            indices(2,2) = 3
            indices(1,3) = 2
            indices(2,3) = 3

            !* check unittest flag to prevent forced symmetrisation of stress matrix
            running_unittest = .true.

            set_type = 1

            !* perform stress calculation for all confs
            call calculate_features_singleset(set_type,.true.,.true.,.false.,.false.,.false.)

            do conf=1,data_sets(set_type)%nconf,1
                stress = data_sets(set_type)%configs(conf)%current_stress
                
                do ii=1,3
                    val1 = stress(indices(1,ii),indices(2,ii))
                    val2 = stress(indices(2,ii),indices(1,ii))

                    if (.not.scalar_equal(val1,val2,dble(1e-10),dble(1e-10),.false.)) then
                        ok = .false.
                    end if
                end do !* end loop over stress matrix diagonals
            end do
            test_stress = ok
        end function test_stress

        logical function test_lookup_tables()
            implicit none
            
            integer :: set_type,conf,atm
            logical,allocatable :: conf_ok(:)
            logical :: parallel,updating_features,scale_features
            real(8),allocatable :: orig_forces(:,:),approx_forces(:,:)
            logical :: pass_test

            scale_features = .false.
            parallel = .false.
            updating_features = .false.
            pass_test = .true.

            do set_type=1,2,1
                if (allocated(conf_ok)) then
                    deallocate(conf_ok)
                end if
                allocate(conf_ok(data_sets(set_type)%nconf))

                do conf=1,data_sets(set_type)%nconf,1
                    if (allocated(dydx)) then
                        deallocate(dydx)
                    end if
                    call allocate_dydx(set_type,conf)
                    call allocate_units(set_type,conf)
                    
                    !* make sure we'are calculating derivatives
                    call switch_property("forces","on")

                    !* make sure no look up tables are being used
                    call switch_performance_option("lookup_tables","off")

                    call deallocate_feature_deriv_info()
                    call calculate_features(scale_features,parallel,updating_features)

                    do atm=1,data_sets(set_type)%configs(conf)%n,1
                        call forward_propagate(set_type,conf)
                        call backward_propagate(set_type,conf)
                    end do
                    
                    if (allocated(orig_forces)) then
                        deallocate(orig_forces)
                    end if
                    allocate(orig_forces(3,data_sets(set_type)%configs(conf)%n))
                    
                    !* analytical forces
                    call calculate_forces(set_type,conf)
                    orig_forces(:,:) = data_sets(set_type)%configs(conf)%current_fi(:,:)
                    
                    
                    !* make sure look up tables are being used
                    call switch_performance_option("lookup_tables","on")

                    call deallocate_feature_deriv_info()
                    call calculate_features(scale_features,parallel,updating_features)

                    
                    do atm=1,data_sets(set_type)%configs(conf)%n,1
                        call forward_propagate(set_type,conf)
                        call backward_propagate(set_type,conf)
                    end do
                    
                    if (allocated(approx_forces)) then
                        deallocate(approx_forces)
                    end if
                    allocate(approx_forces(3,data_sets(set_type)%configs(conf)%n))
                    
                    !* analytical forces
                    call calculate_forces(set_type,conf)
                    approx_forces(:,:) = data_sets(set_type)%configs(conf)%current_fi(:,:)
            
                    if (.not.twoD_array_equal(orig_forces,approx_forces,dble(1e-15),dble(1e-15),&
                    &.false.)) then
                        conf_ok(conf) = .false.
                    else
                        conf_ok(conf) = .true.
                    end if
                end do

                if (.not.all(conf_ok)) then
                    pass_test = .false.
                end if
            end do
            test_lookup_tables = pass_test
        end function test_lookup_tables

        subroutine unittest_error(routine,message)
            implicit none

            character(len=*),intent(in) :: routine,message 
            character,dimension(1:len(routine)+26) :: header 
            header(:) = "*" 
            
            write(*,*) ''
            write(*,*) header
            write(*,*) 'error raised in routine : ',routine
            write(*,*) header
            write(*,*) ''
            write(*,*) 'Error : ',message
            write(*,*) ''
            call exit(0)
        end subroutine unittest_error
end program unittest
