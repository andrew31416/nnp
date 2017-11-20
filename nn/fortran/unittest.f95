program unittest

    use config
    use init
    use propagate
    use util
    use io
    use measures
    use features
    use feature_util

    implicit none

    call main()

    contains
        subroutine main()
            implicit none

            logical :: tests(1:6)
            
            !* net params
            integer :: num_nodes(1:2),nlf_type,fD

            !* features
            integer :: natm,nconf

            !* scratch
            integer :: ii

            integer :: num_tests

            num_tests = size(tests)

            !* number of nodes
            num_nodes(1) = 5
            num_nodes(2) = 3

            !* nonlinear function
            nlf_type = 1
            
            !* features
            fD = 4
            natm = 5
            nconf = 2
            
            call unittest_header()
            
            !* generate atomic info
            call generate_testtrain_set(natm,nconf)

            !* initialise feature memory
            call generate_features(fD)

            !* calculate features and analytical derivatives
            call calculate_features()

            call initialise_net(num_nodes,nlf_type,fD)

            !----------------------!
            !* perform unit tests *!
            !----------------------!
            
            tests(1) = test_dydw()              ! dydw
            call test_loss_jac(tests(2:4))      ! d loss / dw
            tests(5) = test_dydx()              ! dydx
            tests(6) = test_dxdr()              ! d feature / d atom position
            
            do ii=1,num_tests
                call unittest_test(ii,tests(ii))    
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
                    allocate(data_sets(set_type)%configs(conf)%forces(3,natm))
                    
                    data_sets(set_type)%configs(conf)%cell = 0.0d0
                    data_sets(set_type)%configs(conf)%cell(1,1) = 3.d0
                    data_sets(set_type)%configs(conf)%cell(2,1) = 0.1d0
                    data_sets(set_type)%configs(conf)%cell(3,1) = 0.2d0
                    data_sets(set_type)%configs(conf)%cell(1,2) = 0.1d0
                    data_sets(set_type)%configs(conf)%cell(2,2) = 3.0d0
                    data_sets(set_type)%configs(conf)%cell(3,2) = 0.3d0
                    data_sets(set_type)%configs(conf)%cell(1,3) = -0.1d0
                    data_sets(set_type)%configs(conf)%cell(2,3) = 0.2d0
                    data_sets(set_type)%configs(conf)%cell(3,3) = 3.0d0
                    data_sets(set_type)%configs(conf)%n = natm
                    call random_number(data_sets(set_type)%configs(conf)%energy)
                    data_sets(set_type)%configs(conf)%forces = 0.0d0

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
            feature_params%info(2)%ftype = featureID_StringToInt("acsf_behler-g2")            
            feature_params%info(2)%rcut = rcut
            feature_params%info(2)%fs = 0.2d0
            call random_number(feature_params%info(2)%eta)
            call random_number(feature_params%info(2)%za)
            call random_number(feature_params%info(2)%zb)
            
            !* test feature 3
            feature_params%info(3)%ftype = featureID_StringToInt("acsf_normal-iso")
            feature_params%info(3)%rcut = rcut - 1.0d0
            feature_params%info(3)%fs = 0.2d0
            call random_number(feature_params%info(3)%za)
            call random_number(feature_params%info(3)%zb)
            allocate(feature_params%info(3)%prec(1,1))
            call random_number(feature_params%info(3)%prec(1,1)) 
            allocate(feature_params%info(3)%mean(1))
            call random_number(feature_params%info(3)%mean(1)) 

            !* test feature 4
            feature_params%info(4)%ftype = featureID_StringToInt("acsf_behler-g4")
            feature_params%info(4)%rcut = 4.0d0
            feature_params%info(4)%fs = 0.2d0
            call random_number(feature_params%info(4)%lambda)
            call random_number(feature_params%info(4)%xi) 
            call random_number(feature_params%info(4)%eta) 
            call random_number(feature_params%info(4)%za)
            call random_number(feature_params%info(4)%zb)
feature_params%info(4)%lambda = 1.0d0

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

            dy = 0.0d0

            test_result = .false.
                
            !--------------------------!
            !* analytical derivatives *!
            !--------------------------!

            !* forward prop
            call forward_propagate(conf,atm,set_type)
            
            !* back prop
            call backward_propagate(conf,atm,set_type)
          
            !* parse dy/dw into 1d array
            call parse_structure_to_array(dydw,dydw_flat)

            all_ok = .true.

            do ii=1,nwght,1
                w0 = original_weights(ii)
                
                deriv_ok = .false.

                do ww=2,5,1
                
                    dw = 1.0d0/(10.0d0**ww)

                    do jj=1,2,1
                        if (jj.eq.1) then
                            original_weights(ii) = w0 + dw
                        else 
                            original_weights(ii) = w0 - dw
                        end if
                        
                        !* read in weights
                        call parse_array_to_structure(original_weights,net_weights)
                        
                        !* forward propagate
                        call forward_propagate(conf,atm,set_type)

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
                end do
               
                if (deriv_ok.neqv..true.) then
                    all_ok = .false.
                end if    
            end do !* end loop over finite differences
            test_dydw = all_ok
        end function test_dydw

        subroutine test_loss_jac(test_result)
            !* test the jacobian of energy, force, reg. loss functions *!

            implicit none

            logical,intent(out) :: test_result(1:3)

            !* scratch
            integer :: ii,jj,kk,ww,conf,atm,set_type
            real(8) :: dw,w0,dloss,tmp
            real(8),dimension(:),allocatable :: num_jac,anl_jac,original_weights
            logical :: deriv_ok,all_ok

            allocate(num_jac(nwght))
            allocate(anl_jac(nwght))
            allocate(original_weights(nwght))
            
            !* initial weights
            call parse_structure_to_array(net_weights,original_weights)

            conf= 1
            atm = 1
            set_type = 1

            dloss = 0.0d0

            do ii=1,3,1
                if (ii.eq.1) then
                    loss_const_energy = 1.0d0
                    loss_const_forces = 0.0d0
                    loss_const_reglrn = 0.0d0
                else if (ii.eq.2) then
                    loss_const_energy = 0.0d0
                    loss_const_forces = 1.0d0
                    loss_const_reglrn = 0.0d0
                else
                    loss_const_energy = 0.0d0
                    loss_const_forces = 0.0d0
                    loss_const_reglrn = 1.0d0
                end if
                
                !-------------------------------!
                !* analytical jacobian of loss *!
                !-------------------------------!

                call loss_jacobian(original_weights,nwght,set_type,anl_jac)
                
                all_ok = .true.

                do jj=1,nwght,1
                    w0 = original_weights(jj)
                
                    deriv_ok = .false.

                    do ww=2,7,1
                        !* finite difference
                        dw = 1.0d0/(10.0d0**(ww))

                        do kk=1,2,1
                            if (kk.eq.1) then
                                original_weights(jj) = w0 + dw
                            else
                                original_weights(jj) = w0 - dw
                            end if
                            
                            tmp = loss(original_weights,nwght,set_type)

                            if (kk.eq.1) then
                                dloss = tmp
                            else
                                dloss = dloss - tmp
                            end if

                            original_weights(jj) = w0
                        end do !* end loop +/- dw

                        num_jac(jj) = dloss / (2.0d0 * dw)

                        if (scalar_equal(num_jac(jj),anl_jac(jj),dble(1e-7),dble(1e-8),.false.)) then
                            deriv_ok = .true.
                        end if

                    end do !* end loop over +/- dw

                    if (deriv_ok.neqv..true.) then
                        all_ok = .false.
                        write(*,*) ' failing because of',jj
                    end if

                end do !* end loop over weights

                
                test_result(ii) = all_ok
            end do !* end loop over loss terms
        end subroutine test_loss_jac

        logical function test_dydx()
            implicit none

            !* scratch 
            integer :: set_type,conf,atm,xx,ii,ww
            real(8) :: dw,x0
            real(8),allocatable :: num_dydx(:)
            logical,allocatable :: log_atms(:),cnf_atms(:)
            logical :: set_atms(1:2)
            logical :: deriv_ok

            allocate(num_dydx(D))

            do set_type=1,2
                allocate(cnf_atms(data_sets(set_type)%nconf))

                do conf=1,data_sets(set_type)%nconf
                    allocate(log_atms(data_sets(set_type)%configs(conf)%n))
                    
                    do atm=1,data_sets(set_type)%configs(conf)%n
                        log_atms(atm) = .true.

                        !---------------------------!
                        !* analytical differential *!
                        !---------------------------!
                        
                        call forward_propagate(conf,atm,set_type)
                        call backward_propagate(conf,atm,set_type)

                        !--------------------------!
                        !* numerical differential *!
                        !--------------------------!
                        
                        do xx=1,D
                            x0 = data_sets(set_type)%configs(conf)%x(xx+1,atm)

                            deriv_ok = .false.

                            do ww=2,6 
                                !* finite difference for feature
                                dw = 1.0d0/(10**ww)

                                do ii=1,2
                                    if (ii.eq.1) then
                                        data_sets(set_type)%configs(conf)%x(xx+1,atm) = x0 + dw
                                    else
                                        data_sets(set_type)%configs(conf)%x(xx+1,atm) = x0 - dw
                                    end if
                                    
                                    call forward_propagate(conf,atm,set_type)
   
                                    if (ii.eq.1) then
                                        num_dydx(xx) = data_sets(set_type)%configs(conf)%current_ei(atm)
                                    else
                                        num_dydx(xx) = num_dydx(xx) - &
                                                &data_sets(set_type)%configs(conf)%current_ei(atm)
                                    end if
                                    
                                    data_sets(set_type)%configs(conf)%x(xx+1,atm) = x0 
                                end do !* end loop +/- dw

                                num_dydx(xx) = num_dydx(xx) / (2.0d0*dw)

                                if (scalar_equal(num_dydx(xx),dydx(xx),dble(1e-7),dble(1e-10),.false.)) then
                                    deriv_ok = .true.
                                end if
                            end do !* end loop finite differences

                            if (deriv_ok.neqv..true.) then
                                log_atms(atm) = .false.
                            end if
                        end do !* end loop features
                        
                    
                        !log_atms(atm) = array_equal(num_dydx,dydx,dble(1e-7),dble(1e-10),.false.)

                    end do !* end loop atoms
                
                    cnf_atms(conf) = all(log_atms)

                    deallocate(log_atms)
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
            logical :: atom_passes
            

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
                    calc_feature_derivatives = .true.
                    
                    !* calculate analytical derivatives
                    call calculate_features()
                    
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
                    calc_feature_derivatives = .false.

                    do atm=1,data_sets(set_type)%configs(conf)%n
                        do dd=1,3,1
                            !* real space coordinate
                            x0 = data_sets(set_type)%configs(conf)%r(dd,atm)
                           
                            !* if one of finite difference is OK, atom_passes = True
                            atom_passes = .false. 
                            !do ww=3,8,1
                            do ww=5,5,1
        
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
                                    call calculate_features()
                                    
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
                                        if (scalar_equal(num_dxdr(kk,jj),0.0d0,dble(1e-10),dble(1e-10),.false.)&
                                        &.neqv..true.) then
                                            deriv_matches = .false.
                                            do ll=1,anl_deriv(kk,jj)%n
                                                !* loop over atoms which contribute to feature kk,atom jj
                                                if (anl_deriv(kk,jj)%idx(ll).eq.atm) then 
                                                    if ( scalar_equal(num_dxdr(kk,jj),&
                                                    &anl_deriv(kk,jj)%vec(dd,ll),dble(1e-7),&
                                                    &dble(1e-7),.true.) ) then
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

end program unittest
